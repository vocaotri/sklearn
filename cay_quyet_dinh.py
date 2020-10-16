from sklearn import tree

# thup thap data
# xu ly data
# xay dung model
# du doan ket qua
# danh gia
my_tree = tree.DecisionTreeClassifier()
dattrung = [
    [1, 3, 3, 7],
    [5, 2, 4, 6],
    [1, 2, 4, 6],
    [5, 4, 4, 3],
    [3, 2, 3, 7],
    [3, 3, 3, 6],
    [5, 2, 2, 7],
    [1, 2, 4, 3]
]
nhan = [0, 1, 1, 0, 0, 0, 0, 1]

result = my_tree.fit(dattrung, nhan)

kq = result.predict([[1, 2, 4, 6]])
print(kq)
