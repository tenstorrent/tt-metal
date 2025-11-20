#!/bin/bash
#
# Generate a JSON list of files changed in a given commit.
# Usage:
#   ./get_changed_files.sh <commit_hash> [output_path]
# Default output: auto_triage/data/commit_<hash>_files.json

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <commit_hash> [output_path]" >&2
    exit 1
fi

COMMIT="$1"
OUTPUT_PATH="${2:-auto_triage/data/commit_${COMMIT}_files.json}"

if ! git cat-file -e "${COMMIT}^{commit}" >/dev/null 2>&1; then
    echo "Error: commit '${COMMIT}' not found." >&2
    exit 1
fi

DATA_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$DATA_DIR"
rm -f "$OUTPUT_PATH"

TMP_FILE="$(mktemp)"
trap 'rm -f "$TMP_FILE"' EXIT

git diff-tree --no-commit-id --name-status -r "$COMMIT" > "$TMP_FILE"

jq -R -s '
    split("\n")
    | map(select(length > 0) | split("\t"))
    | map({
        status: (.[0] // ""),
        file: (.[1] // "")
      })
' "$TMP_FILE" > "$OUTPUT_PATH"

echo "Changed files saved to: $OUTPUT_PATH"
