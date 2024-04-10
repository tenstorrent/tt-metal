# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import hashlib
import github
from github import Github

# GitHub

# Testing repo
"""
REPO_OWNER = "xanderchin"
REPO_NAME = "test_github_api"
"""
REPO_OWNER = "tenstorrent-metal"
REPO_NAME = "tt-metal"

# Params
BOT_LABEL_COL = "900C3F"


def generate_sha1_hash(input_string):
    # Convert the input string to bytes
    input_bytes = input_string.encode("utf-8")

    # Create a new SHA-1 hash object
    sha1_hash = hashlib.sha1()

    # Update the hash object with the input bytes
    sha1_hash.update(input_bytes)

    # Get the hexadecimal representation of the hash
    hex_hash = sha1_hash.hexdigest()

    return hex_hash


def open_repo():
    api_key = os.environ.get("GITHUB_API_KEY")
    g = Github(api_key)
    repo = g.get_repo(f"{REPO_OWNER}/{REPO_NAME}")
    return repo


def create_or_get_label(repo, unique_reproduction_cmd_or_pytest):
    name = f":robot:Issue:{generate_sha1_hash(unique_reproduction_cmd_or_pytest)}"
    name = name[:50]
    try:
        label = repo.get_label(name)
    except github.GithubException as e:
        """
        github.GithubException.UnknownObjectException
        print(e)
        """
        label = repo.create_label(name, BOT_LABEL_COL)

    return label


def find_open_issues(repo, label):
    list_of_issues = repo.get_issues(state="open", labels=[label])
    return list_of_issues


def create_or_update_issue(repo, *, full_test_name, body, assignees=[], additional_labels=[]):
    unique_label = create_or_get_label(repo, full_test_name)

    issues = find_open_issues(repo, unique_label)
    if issues.totalCount == 0:
        title = f":robot:AutomaticIssue Failure of '{full_test_name}'"
        labels = [unique_label] + additional_labels
        issue = repo.create_issue(title=title, body=body, labels=labels, assignees=assignees)
    else:
        # Only return the first issue
        issue = issues[0]
        # But update all issues
        for i in issues:
            i.create_comment(body)

    return


def create_or_update_pytest_issue(pytest_gh_issue):
    if "GITHUB_API_KEY" not in os.environ:
        import pprint

        print("No GITHUB_API_KEY in envs. github_automatic_issue creation in debug mode")
        pprint.pprint(pytest_gh_issue)
        return

    repo = open_repo()
    create_or_update_issue(repo, **pytest_gh_issue)


import sys
from loguru import logger
import pytest


@pytest.mark.parametrize("test_param", ["this_parameter_fails", "this_parameter_passes"])
def test_issue_creation(test_param, github_issue):
    # logger.add(sys.stderr)
    # logger.add(sys.stdout)
    # print(f"this test is the test with '{test_param}'")
    # logger.debug("this is debug text")
    github_issue["body"] = f"Test failure for test_param: '{test_param}'"
    github_issue["assignees"] = ["xanderchin"]
    github_issue["additional_labels"] = [":robot:Issue"]
    assert test_param == "this_parameter_passes"
