# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import re

"""
Commit msg must be in the format:
"""
FORMAT_MSG = "#<GH ISSUE NUMBER> or MET-<JIRA ISSUE NUMBER>: <non-empty message>"
VALID_PREFIXES = "|".join(["#", "MET-"])
MATCHING_REGEX = f"^({VALID_PREFIXES})(\d+\:\ .)"


def print_commit_msg(commit_msg_whole):
    print("\t --- Printing commit message ---")
    print(commit_msg_whole)


if __name__ == "__main__":
    argv = sys.argv
    argc = len(sys.argv)

    assert argc == 2, f"Expected two arguments"

    commit_msg_filename = argv[1]

    with open(commit_msg_filename) as commit_msg_file:
        commit_msg_whole = commit_msg_file.read()

    search_result = re.search(MATCHING_REGEX, commit_msg_whole)

    if not search_result:
        print_commit_msg(commit_msg_whole)
        raise Exception(f"Commit message does match format {FORMAT_MSG}")

    result_groups = search_result.groups(default=tuple())

    if not result_groups:
        print_commit_msg(commit_msg_whole)
        raise Exception("Regex matching error during commit message sanitization")
