#!/bin/bash

# Define function to display usage information
usage() {
    echo "Usage: $0 -o <owner> -r <repo> -c <commit_hash> -t <github_token>"
    echo "Options:"
    echo "  -o  GitHub owner (user or organization name)"
    echo "  -r  GitHub repository name"
    echo "  -c  Exact 40-character commit hash (SHA)"
    echo "  -t  GitHub Personal Access Token (PAT) with 'actions:read' scope"
    exit 1
}

# Parse command-line arguments using getopts
while getopts "o:r:c:t:" opt; do
    case "$opt" in
        o) OWNER="$OPTARG" ;;
        r) REPO="$OPTARG" ;;
        c) COMMIT_HASH="$OPTARG" ;;
        t) GITHUB_TOKEN="$OPTARG" ;;
        *) usage ;;
    esac
done

# Check if all required arguments were provided
if [ -z "$OWNER" ] || [ -z "$REPO" ] || [ -z "$COMMIT_HASH" ] || [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: Missing required arguments."
    usage
fi

# --- API Call and Parsing ---
API_URL="https://api.github.com/repos/$OWNER/$REPO/actions/runs?head_sha=$COMMIT_HASH&per_page=1"

RUN_ID=$(curl -s -L \
    -H "Accept: application/vnd.github.v3+json" \
    -H "Authorization: token $GITHUB_TOKEN" \
    "$API_URL" | \
    jq -r '.workflow_runs[0].id')

# --- Output ---
if [[ "$RUN_ID" != "null" && -n "$RUN_ID" ]]; then
    echo "$RUN_ID" # Print only the ID for easy use in other scripts
    # You can uncomment the line below for a more verbose output:
    # echo "Workflow Run ID for commit $COMMIT_HASH: $RUN_ID"
else
    # Output an empty string and a message to stderr for scripting reliability
    echo >&2 "Error: No workflow run found for commit $COMMIT_HASH on $OWNER/$REPO."
fi
