#!/bin/bash
set -euo pipefail
# Wrapper script for clang-tidy that exports fixes to unique YAML files.
# Usage: clang-tidy-export-fixes-wrapper.sh <fixes-dir> <clang-tidy-args...>
#
# Exports per-TU fix files so parallel builds don't clobber each other.
# Uses clang-tidy-cache (ctcache) when CTCACHE_S3_OK=1 is set in the environment.

FIXES_DIR="$1"
shift
CLANG_TIDY="clang-tidy-23"

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
    exec "$CLANG_TIDY" "$@"
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
    exec "$CTCACHE_BIN" "$CLANG_TIDY" --export-fixes="$FIXES_FILE" "$@"
elif [ -n "${CTCACHE_HOST:-}" ] && [ -n "${CTCACHE_BIN:-}" ]; then
    # TorrentCI: the platform renders CTCACHE_HOST (its loopback cache proxy)
    # only for jobs that opt in with `analysis-cache: ctcache`. ctcache stores
    # a marker only when clang-tidy exits 0 with empty stdout, so a TU that
    # exits 0 but prints something is never cached; log one line per TU
    # ("<exit> <stdout non-empty>") beside the fixes dir, never in it, so the
    # job can count those TUs.
    OUT=$(mktemp)
    rc=0
    "$CTCACHE_BIN" "$CLANG_TIDY" --export-fixes="$FIXES_FILE" "$@" >"$OUT" || rc=$?
    cat "$OUT"
    printf '%d %d\n' "$rc" "$([ -s "$OUT" ] && echo 1 || echo 0)" >>"${FIXES_DIR%/}.ctcache-tus"
    rm -f "$OUT"
    exit "$rc"
else
    exec "$CLANG_TIDY" --export-fixes="$FIXES_FILE" "$@"
fi
