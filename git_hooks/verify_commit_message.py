import sys
import re

"""
Commit msg must be in the format:
"""
FORMAT_MSG = "#<GH ISSUE NUMBER>: <non-empty message>"
MATCHING_REGEX = "^(#\d+\:\ .)"


if __name__ == "__main__":
    argv = sys.argv
    argc = len(sys.argv)

    assert argc == 2, f"Expected two arguments"

    commit_msg_filename = argv[1]

    with open(commit_msg_filename) as commit_msg_file:
        commit_msg_whole = commit_msg_file.read()

    search_result = re.search(MATCHING_REGEX, commit_msg_whole)

    if not search_result:
        raise Exception(f"Commit message does match format {FORMAT_MSG}")

    result_groups = search_result.groups(default=tuple())

    if not result_groups:
        raise Exception("Regex matching error during commit message sanitization")
