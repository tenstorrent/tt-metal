# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import argparse
import requests

# The following script requires a github personal access token.
# See: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token


def get_pending_reviewers(pull_numbers, github_token):
    owner = "tenstorrent-metal"
    repo = "tt-metal"
    headers = {"Authorization": f"token {github_token}"}
    pending_reviewers = set()

    for pull_number in pull_numbers:
        requested_reviewers_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/requested_reviewers"
        reviews_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/reviews"

        requested_reviewers_response = requests.get(requested_reviewers_url, headers=headers)
        requested_reviewers = (
            {reviewer["login"] for reviewer in requested_reviewers_response.json().get("users", [])}
            if requested_reviewers_response.status_code == 200
            else set()
        )

        reviews_response = requests.get(reviews_url, headers=headers)
        reviews = reviews_response.json() if reviews_response.status_code == 200 else []

        if any(review["state"] == "APPROVED" for review in reviews):
            continue

        pending_reviewers.update(requested_reviewers - {review["user"]["login"] for review in reviews})

    return pending_reviewers


def get_reviewers(pull_numbers, github_token):
    owner = "tenstorrent-metal"
    repo = "tt-metal"
    reviewers = set()
    headers = {"Authorization": f"token {github_token}"}

    for pull_number in pull_numbers:
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/requested_reviewers"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            for reviewer in data["users"]:
                reviewers.add(reviewer["login"])
        else:
            print(f"Failed to get reviewers for PR {pull_number}: {response.status_code}")

    return list(reviewers)


def get_my_pending_reviews(username, github_token):
    headers = {"Authorization": f"token {github_token}"}
    pending_reviews_urls = []

    prs_url = f"https://api.github.com/search/issues?q=is:pr+review-requested:{username}&sort=created&order=desc"
    prs_response = requests.get(prs_url, headers=headers)
    prs = prs_response.json().get("items", []) if prs_response.status_code == 200 else []

    for pr in prs:
        repo = "/".join(pr["repository_url"].split("/")[-2:])
        pull_number = pr["number"]
        pr_url = pr["html_url"]
        pr_title = pr["title"]

        requested_reviewers_url = f"https://api.github.com/repos/{repo}/pulls/{pull_number}/requested_reviewers"
        reviews_url = f"https://api.github.com/repos/{repo}/pulls/{pull_number}/reviews"

        requested_reviewers_response = requests.get(requested_reviewers_url, headers=headers)
        requested_reviewers = (
            {reviewer["login"] for reviewer in requested_reviewers_response.json().get("users", [])}
            if requested_reviewers_response.status_code == 200
            else set()
        )

        reviews_response = requests.get(reviews_url, headers=headers)
        reviews = reviews_response.json() if reviews_response.status_code == 200 else []

        if username in requested_reviewers and not any(review["state"] == "APPROVED" for review in reviews):
            pending_reviews_urls.append((pr_title, pr_url))

    return pending_reviews_urls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token")
    parser.add_argument("--issues", nargs="+", type=int, help="List of pull request numbers")
    parser.add_argument("--username")

    token = parser.parse_args().token
    issues = parser.parse_args().issues
    username = parser.parse_args().username

    if token == None:
        print("You forget to provide the --token")

    if issues != None:
        reviewers = get_reviewers(issues, token)
        print("Reviewers:", reviewers)

        pending_reviewers = get_pending_reviewers(issues, token)
        print("Pending Reviewers:", pending_reviewers)

    elif username != None:
        pending_reviews = get_my_pending_reviews(username, token)
        for title, url in pending_reviews:
            print(f"{url} -> {title}")

    else:
        print("You need to specify either the --username or the --issues")


if __name__ == "__main__":
    main()
