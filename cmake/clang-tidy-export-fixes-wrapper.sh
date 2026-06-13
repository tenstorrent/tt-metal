#!/bin/bash
set -euo pipefail
# Wrapper script for clang-tidy that exports fixes to unique YAML files.
# Usage: clang-tidy-export-fixes-wrapper.sh <fixes-dir> <clang-tidy-args...>
#
# Exports per-TU fix files so parallel builds don't clobber each other.
# Uses clang-tidy-cache (ctcache) when CTCACHE_S3_OK=1 is set in the environment.

FIXES_DIR="$1"
shift

mkdir -p "$FIXES_DIR"

# Find the source file (last non-flag arg before --)
SOURCE_FILE=""
for arg in "$@"; do
    if [[ "$arg" != -* ]]; then
        SOURCE_FILE="$arg"
    fi
done

if [[ -z "$SOURCE_FILE" ]]; then
    exec clang-tidy-20 "$@"
fi

SOURCE_FILE_ABS=$(realpath "$SOURCE_FILE")
HASH=$(echo "$SOURCE_FILE_ABS" | md5sum | cut -d' ' -f1)
BASENAME=$(basename "$SOURCE_FILE")
FIXES_FILE="${FIXES_DIR}/${BASENAME}-${HASH}.yaml"

if command -v clang-tidy-cache &>/dev/null && [ "${CTCACHE_S3_OK:-0}" = "1" ]; then
    exec clang-tidy-cache clang-tidy-20 --export-fixes="$FIXES_FILE" "$@"
else
    exec clang-tidy-20 --export-fixes="$FIXES_FILE" "$@"
fi
