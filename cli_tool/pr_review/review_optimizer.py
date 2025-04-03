#!/usr/bin/env python

import argparse
import subprocess
import json
import re
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Get list of reviewers for a PR.')
    parser.add_argument('pr_number', type=int, help='Pull request number')
    parser.add_argument('-include', type=str, nargs='*', default=[], help='List of names to include')
    parser.add_argument('-skip', type=str, nargs='*', default=[], help='List of names to skip')
    return parser.parse_args()

def get_pr_files(pr_number, repo):
    try:
        result = subprocess.run(
            ['gh', 'pr', 'view', str(pr_number), '--repo', repo, '--json', 'files'],
            check=True, text=True, capture_output=True
        )
        data = json.loads(result.stdout)
        return data['files']
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error fetching PR files: {e.stderr}")

def get_codeowners(file_path):
    """Get codeowners for a file path"""
    try:
        result = subprocess.run(
            ["codeowners", file_path],
            capture_output=True,
            text=True,
            check=True
        )
        # "file_path  @owner1 @owner2 ..."
        output = result.stdout.strip()
        
        # add the owners to a list
        if output and file_path in output:
            owners_part = output.replace(file_path, "").strip()
            owners_list = owners_part.split() if owners_part else []
        else:
            owners_list = []
        # print(f"Owners for {file_path}: {owners_list}")
        return owners_list
    except subprocess.CalledProcessError as e:
        print(f"Error getting codeowners for {file_path}: {e}")
        return []

def select_reviewers(file_owners, include, skip):
    # Normalize inputs to strip '@' from include and skip lists
    include = {name.lstrip('@') for name in include}
    skip = {name.lstrip('@') for name in skip}
    
    # must include reviewers and reviewers can be skipped (on vacation, etc.)
    selected_reviewers = set(include)

    for file in file_owners:
        file['owners'] = [owner.lstrip('@') for owner in file['owners'] if owner.lstrip('@') not in skip]

    # Mark files covered by the included reviewers as reviewed
    for file in file_owners:
        if any(owner in include for owner in file['owners']):
            file['owners'] = []

    # Refreshes untouched files after including reviewers
    untouched_files = [file for file in file_owners if file['owners']]

    owner_files = {}
    for file in untouched_files:
        for owner in file['owners']:
            if owner in owner_files:
                owner_files[owner].add(file['filename'])
            else:
                owner_files[owner] = {file['filename']}

    # Sort the owners by the number of unique files they cover
    while untouched_files:
        max_coverage_owner = max(owner_files, key=lambda o: len(owner_files[o]), default=None)
        
        if not max_coverage_owner:
            break
        
        selected_reviewers.add(max_coverage_owner.lstrip('@'))  # Normalize before adding
        
        # Remove all files that the selected owner can review
        covered_files = owner_files.pop(max_coverage_owner, set())
        untouched_files = [file for file in untouched_files if file['filename'] not in covered_files]

        # Update owner_files by removing the files already reviewed
        for owner in owner_files:
            owner_files[owner] -= covered_files

    return sorted(f"@{reviewer}" for reviewer in selected_reviewers)  # Ensure consistent output format



def review_optimizer():
    args = parse_args()
    repo = "tenstorrent/tt-metal"

    try:
        files = get_pr_files(args.pr_number, repo)

        file_owners = [
            {"filename": file['path'], "owners": get_codeowners(file['path'])}
            for file in files
        ]
        reviewers = select_reviewers(file_owners, args.include, args.skip)
        print("Selected Reviewers:", reviewers)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    review_optimizer()

