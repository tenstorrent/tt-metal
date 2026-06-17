import sys, pytest

ids = []


class C:
    def pytest_collection_modifyitems(self, items):
        for it in items:
            ids.append(it.nodeid)


pytest.main(["--collect-only", "-p", "no:cacheprovider", "-q", sys.argv[1]], plugins=[C()])
with open(sys.argv[2], "w") as f:
    f.write("\n".join(ids) + "\n")
print("WROTE", len(ids))
