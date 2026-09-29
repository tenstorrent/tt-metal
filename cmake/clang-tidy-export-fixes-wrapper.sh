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

# Source file = last non-flag arg before the compiler '--' separator.
# Break at '--' to avoid grabbing post-separator args as the source file.
SOURCE_FILE=""
for arg in "$@"; do
    [[ "$arg" == "--" ]] && break
    [[ "$arg" == -* ]] && continue
    SOURCE_FILE="$arg"
done

if [[ -z "$SOURCE_FILE" ]]; then
    exec clang-tidy-20 "$@"
fi

# Time-budget drain. When CLANG_TIDY_DEADLINE_EPOCH (unix seconds) has passed, fail
# fast instead of analyzing. CMake skips the real compile when the tidy launcher
# fails, so ninja (-k0) drains the remaining edges in seconds while in-flight TUs
# finish and get written to ctcache. Without this, the CI step timeout kills the
# build mid-flight and those TUs are lost. CLANG_TIDY_DEADLINE_MARKER, if set, is
# touched so the workflow can tell "ran out of time" apart from real failures.
if [[ -n "${CLANG_TIDY_DEADLINE_EPOCH:-}" ]] && (( $(printf '%(%s)T' -1) >= CLANG_TIDY_DEADLINE_EPOCH )); then
    echo "clang-tidy: time budget exhausted, skipping ${SOURCE_FILE}" >&2
    [[ -n "${CLANG_TIDY_DEADLINE_MARKER:-}" ]] && : > "$CLANG_TIDY_DEADLINE_MARKER"
    exit 1
fi

# Unique, collision-free suffix from the path.
# Uses one fork (md5sum); basename and hash trimming are bash builtins.
BASENAME="${SOURCE_FILE##*/}"
HASH=$(md5sum <<<"$SOURCE_FILE")
HASH="${HASH%% *}"
FIXES_FILE="${FIXES_DIR}/${BASENAME}-${HASH}.yaml"

# CTCACHE_BIN is pre-resolved once by the workflow step that sets CTCACHE_S3_OK,
# avoiding a repeated PATH scan across thousands of TU invocations.
if [ "${CTCACHE_S3_OK:-0}" = "1" ] && [ -n "${CTCACHE_BIN:-}" ]; then
    exec "$CTCACHE_BIN" clang-tidy-20 --export-fixes="$FIXES_FILE" "$@"
else
    exec clang-tidy-20 --export-fixes="$FIXES_FILE" "$@"
fi
