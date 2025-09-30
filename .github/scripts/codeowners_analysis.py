#!/usr/bin/env python3
"""
CodeOwners Analysis Script

This script analyzes CODEOWNERS file and determines which teams and individuals
need to approve changes in a PR. It resolves GitHub usernames to full names
and outputs the results in a format suitable for GitHub workflow consumption.
"""

import os
import sys
import json
import urllib.request
from urllib.error import HTTPError

# Import codeowners package (required)
try:
    from codeowners import CodeOwners
except ImportError:
    print("Error: codeowners package is required but not installed.")
    print("Install it with: pip install codeowners")
    sys.exit(1)


def get_user_full_name(username):
    """Get full name for a GitHub username using GitHub API."""
    if not username or not username.startswith("@"):
        return username

    # Remove @ prefix for API call
    clean_username = username[1:]  # Remove @

    # Skip API calls for team names (containing /)
    if "/" in clean_username:
        return clean_username

    # Try both tokens
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("ORG_READ_GITHUB_TOKEN") or ""

    if not token:
        print(f"Warning: No token available for user lookup of {username}")
        return clean_username

    try:
        # Use GitHub API to get user information
        url = f"https://api.github.com/users/{clean_username}"
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/vnd.github.v3+json")
        req.add_header("User-Agent", "GitHub-Actions-CodeOwners-Analysis")

        with urllib.request.urlopen(req) as response:
            if response.getcode() == 200:
                user_data = json.loads(response.read().decode())
                return user_data.get("name") or clean_username
            else:
                print(f"Warning: API returned {response.getcode()} for user {username}")
                return clean_username
    except HTTPError as e:
        if e.code == 401:
            print(f"Warning: Unauthorized access for user {username} (insufficient token permissions)")
        elif e.code == 403:
            print(f"Warning: Forbidden access for user {username} (token lacks user:read scope)")
        elif e.code == 404:
            print(f"Warning: User {username} not found")
        else:
            print(f"Warning: HTTP error {e.code} for user {username}")
        return clean_username
    except Exception as e:
        print(f"Warning: Error getting name for {username}: {e}")
        return clean_username


def analyze_codeowners(changed_files_path, codeowners_path):
    """Analyze CODEOWNERS file and return required groups using codeowners package."""

    # Read changed files
    with open(changed_files_path, "r") as f:
        changed_files = [line.strip() for line in f if line.strip()]

    print(f"Analyzing {len(changed_files)} changed files using codeowners package...")

    # Use codeowners package - read file content first
    with open(codeowners_path, "r") as f:
        codeowners_content = f.read()

    co = CodeOwners(codeowners_content)

    # Parse CODEOWNERS file - find all unique patterns that match changed files
    # and collect their owners
    pattern_groups = {}  # pattern -> set of (username, full_name) tuples
    team_groups = set()
    processed_patterns = set()  # Track patterns we've already processed

    # First pass: find all unique patterns that match any changed files
    for file_path in changed_files:
        matching_lines = list(co.matching_lines(file_path))
        if matching_lines:
            # GitHub CODEOWNERS precedence: last matching pattern takes precedence
            # Sort by line number (highest first) and use only the most specific match
            sorted_matches = sorted(matching_lines, key=lambda x: x[1], reverse=True)
            best_match = sorted_matches[0]

            print(
                f"Found {len(matching_lines)} matching lines for {file_path}, using most specific (line {best_match[1]})"
            )

            # Use only the owners from the most specific match
            if len(best_match) >= 3:
                owners_list = best_match[0]  # First element is the owners list
                pattern = best_match[2]  # Third element is the pattern

                # Only process this pattern if we haven't seen it before
                if pattern not in processed_patterns:
                    processed_patterns.add(pattern)

                    if pattern not in pattern_groups:
                        pattern_groups[pattern] = set()

                    for owner_type, owner in owners_list:
                        if owner_type == "TEAM":
                            # This is a team
                            team_groups.add(owner)
                        elif owner_type in ["USERNAME", "EMAIL"]:
                            # This is an individual - get full name and store both username and full name
                            full_name = get_user_full_name(owner)
                            # Store as tuple: (username, full_name)
                            username = owner[1:] if owner.startswith("@") else owner  # Remove @ prefix if present
                            pattern_groups[pattern].add((username, full_name))
        else:
            print(f"No matches found for {file_path}")

    # Create output - teams are separate, individuals are grouped by pattern
    teams_list = ",".join(sorted(team_groups))

    # For individuals, we need to group them by their patterns
    # Each pattern becomes a "group" that requires approval from any of its members
    pattern_groups_list = []
    for pattern, owners in pattern_groups.items():
        if owners:  # Only include patterns that have individuals
            # Format: pattern:username1|full_name1,username2|full_name2,... (username|full_name pairs)
            owners_pairs = []
            for username, full_name in sorted(owners, key=lambda x: x[1]):  # Sort by full name
                owners_pairs.append(f"{username}|{full_name}")
            owners_str = ",".join(owners_pairs)
            pattern_groups_list.append(f"{pattern}:{owners_str}")

    individuals_list = "@@@".join(pattern_groups_list) if pattern_groups_list else ""

    # Combine all groups
    if teams_list and individuals_list:
        all_groups = teams_list + "," + individuals_list
    elif teams_list:
        all_groups = teams_list
    else:
        all_groups = individuals_list

    print(f"Found {len(team_groups)} team groups and {len(pattern_groups)} pattern groups")
    print(f"Teams: {teams_list}")
    print(f"Individuals: {individuals_list}")

    return {
        "all_groups": all_groups,
        "teams": teams_list,
        "individuals": individuals_list,
        "changed_files": "\n".join(changed_files),
    }


def main():
    """Main entry point when run as script."""
    if len(sys.argv) != 3:
        print("Usage: python codeowners_analysis.py <changed_files.txt> <codeowners_path>")
        sys.exit(1)

    changed_files_path = sys.argv[1]
    codeowners_path = sys.argv[2]

    if not os.path.exists(changed_files_path):
        print(f"Error: Changed files path '{changed_files_path}' does not exist")
        sys.exit(1)

    if not os.path.exists(codeowners_path):
        print(f"Error: CODEOWNERS path '{codeowners_path}' does not exist")
        sys.exit(1)

    result = analyze_codeowners(changed_files_path, codeowners_path)

    # Output in GitHub Actions format if GITHUB_OUTPUT is available
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        # Clear the file first, then write
        with open(github_output, "w") as f:
            f.write("changed-files<<EOF\n")
            f.write(result["changed_files"])
            f.write("\nEOF\n")

            f.write("codeowners-groups<<EOF\n")
            f.write(result["all_groups"])
            f.write("\nEOF\n")

            f.write("codeowners-teams<<EOF\n")
            f.write(result["teams"])
            f.write("\nEOF\n")

            f.write("codeowners-individuals<<EOF\n")
            f.write(result["individuals"])
            f.write("\nEOF\n")
    else:
        # When run outside GitHub Actions, print results to stdout
        print("\n" + "=" * 50)
        print("RESULTS (for GitHub Actions):")
        print("=" * 50)
        print(f"changed-files={repr(result['changed_files'])}")
        print(f"codeowners-groups={repr(result['all_groups'])}")
        print(f"codeowners-teams={repr(result['teams'])}")
        print(f"codeowners-individuals={repr(result['individuals'])}")
        print("=" * 50)


if __name__ == "__main__":
    main()
