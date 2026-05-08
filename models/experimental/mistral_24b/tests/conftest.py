# try:
#     from tracy import signpost
# except ImportError:
#     def signpost(*args, **kwargs):
#         pass


# def pytest_runtest_setup(item):
#     signpost("Mistral24B::TestStart", item.nodeid)


# def pytest_runtest_teardown(item, nextitem):
#     signpost("Mistral24B::TestEnd", item.nodeid)
