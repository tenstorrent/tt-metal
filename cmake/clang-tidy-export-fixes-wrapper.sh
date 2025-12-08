#!/bin/bash
set -euo pipefail
# Wrapper script for clang-tidy that exports fixes to unique YAML files
# Usage: clang-tidy-export-fixes-wrapper.sh <fixes-dir> <clang-tidy-args...>
#
# This script invokes clang-tidy and exports fixes to a unique file based on
# the hash of the source file path, allowing parallel execution without conflicts.

FIXES_DIR="$1"
shift

# Create the fixes directory if it doesn't exist
mkdir -p "$FIXES_DIR"

# Find the source file from the arguments (last argument that's a file)
SOURCE_FILE=""
for arg in "$@"; do
    if [[ "$arg" != -* ]]; then
        SOURCE_FILE="$arg"
    fi
done

if [[ -z "$SOURCE_FILE" ]]; then
    # If no source file found, just run clang-tidy without export-fixes
    exec clang-tidy-20 "$@"
fi

# Generate a unique filename based on the absolute source file path
SOURCE_FILE_ABS=$(realpath "$SOURCE_FILE")
HASH=$(echo "$SOURCE_FILE_ABS" | md5sum | cut -d' ' -f1)
BASENAME=$(basename "$SOURCE_FILE")
FIXES_FILE="${FIXES_DIR}/${BASENAME}-${HASH}.yaml"

exec clang-tidy-20 --export-fixes="$FIXES_FILE" "$@"
