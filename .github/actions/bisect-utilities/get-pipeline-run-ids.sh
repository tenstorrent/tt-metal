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
    echo "Error: Missing required arguments." >&2
    usage
fi

# --- API Call and Parsing ---
# Note: Removed 'per_page=1' to fetch all runs. The default 'per_page' is 30,
# you may need to implement pagination for repos with many runs per commit.
API_URL="https://api.github.com/repos/$OWNER/$REPO/actions/runs?head_sha=$COMMIT_HASH"

# Curl the API and use jq to extract the 'id' field from every object in the
# 'workflow_runs' array. The -r flag ensures raw (unquoted) output.
RUN_IDS=$(curl -s -L \
    -H "Accept: application/vnd.github.v3+json" \
    -H "Authorization: token $GITHUB_TOKEN" \
    "$API_URL" | \
    jq -r '.workflow_runs[].id')

# --- Output ---
if [[ -n "$RUN_IDS" ]]; then
    echo "$RUN_IDS" # Prints all IDs, one per line
else
    # Output an error message to stderr
    echo "Error: No workflow runs found for commit $COMMIT_HASH on $OWNER/$REPO." >&2
fi
