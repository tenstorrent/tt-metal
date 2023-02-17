import sys

if __name__ == "__main__":
    argv = sys.argv
    argc = len(sys.argv)

    assert argc == 2, f"Expected two arguments"

    commit_msg_filename = argv[1]

    for line in open(commit_msg_filename):
        pass
