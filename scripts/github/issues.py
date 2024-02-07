# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import argparse
import requests

# The following script requires a github personal access token.
# See: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token


def get_reviewers_and_pending_reviewers(pull_numbers, github_token):
    owner = "tenstorrent-metal"
    repo = "tt-metal"
    headers = {"Authorization": f"token {github_token}"}
    reviewers_info = {}

    for pull_number in pull_numbers:
        requested_reviewers_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/requested_reviewers"
        reviews_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/reviews"

        requested_reviewers_response = requests.get(requested_reviewers_url, headers=headers)
        if requested_reviewers_response.status_code == 200:
            reviewers = {reviewer["login"] for reviewer in requested_reviewers_response.json().get("users", [])}
        else:
            print(
                f"Unable to find Requested Reviews for PR {pull_number}: HTTP {requested_reviewers_response.status_code} -> {requested_reviewers_response.text}"
            )
            continue

        reviews_response = requests.get(reviews_url, headers=headers)
        if reviews_response.status_code == 200:
            reviews = reviews_response.json()
        else:
            print(
                f"Unable to find Reviews for PR {pull_number}: HTTP {reviews_response.status_code} -> {requested_reviewers_response.text}"
            )
            continue

        pr_details_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"
        pr_details_response = requests.get(pr_details_url, headers=headers)
        if pr_details_response.status_code == 200:
            pr_data = pr_details_response.json()
            base_branch = pr_data.get("base", {}).get("ref", "")
            state = pr_data.get("state")
            mergeable = pr_data.get("mergeable")
        else:
            print(f"Unable to find details for PR {pull_number}: HTTP {pr_details_response.status_code}")
            continue

        pending_reviewers = reviewers - {
            review["user"]["login"] for review in reviews if review["state"] in ["APPROVED", "CHANGES_REQUESTED"]
        }

        reviewers_info[pull_number] = {
            "reviewers": reviewers,
            "pending_reviewers": pending_reviewers,
            "state": state,
            "mergeable": mergeable,
        }

    return reviewers_info


def get_my_pending_reviews(username, github_token):
    headers = {"Authorization": f"token {github_token}"}
    pending_reviews_urls = []

    prs_url = (
        f"https://api.github.com/search/issues?q=is:pr+is:open+review-requested:{username}&sort=created&order=desc"
    )
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
        reviewers_info = get_reviewers_and_pending_reviewers(issues, token)

        for pr, info in reviewers_info.items():
            print(
                f"PR {pr} - Reviewers: {info['reviewers']}, State: {info['state']}, Pending Reviewers: {info['pending_reviewers']}, Mergeable: {info['mergeable']}"
            )

    elif username != None:
        pending_reviews = get_my_pending_reviews(username, token)
        for title, url in pending_reviews:
            print(f"{url} -> {title}")

    else:
        print("You need to specify either the --username or the --issues")


if __name__ == "__main__":
    main()
