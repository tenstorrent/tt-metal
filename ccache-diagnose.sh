#!/usr/bin/env bash
# ccache-diagnose.sh — targeted at tt_metal/hal.cpp
set -euo pipefail

BUILD_DIR="${1:?Usage: $0 <cmake-build-dir>}"

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Find the specific entry for hal.cpp
ENTRY=$(jq 'map(select(.file | test("tt_metal/hal\\.cpp$"))) | first' \
  "$BUILD_DIR/compile_commands.json")

if [[ "$ENTRY" == "null" || -z "$ENTRY" ]]; then
  echo "ERROR: tt_metal/hal.cpp not found in compile_commands.json"
  exit 1
fi

FILE=$(echo "$ENTRY" | jq -r '.file')
DIR=$(echo "$ENTRY"  | jq -r '.directory')
CMD=$(echo "$ENTRY"  | jq -r '.command')

echo "========================================"
echo "SECTION 1: ccache config"
echo "========================================"
ccache --show-config

echo ""
echo "========================================"
echo "SECTION 2: compiler identity"
echo "========================================"
which clang clang++ 2>/dev/null || true
clang --version 2>/dev/null || true
sha256sum $(which clang) $(which clang++) 2>/dev/null || true

echo ""
echo "========================================"
echo "SECTION 3: system include paths"
echo "========================================"
clang -v -x c++ /dev/null -fsyntax-only 2>&1 || true

echo ""
echo "========================================"
echo "SECTION 4: source file and working dir"
echo "========================================"
echo "file: $FILE"
echo "dir:  $DIR"

echo ""
echo "========================================"
echo "SECTION 5: compile command"
echo "========================================"
echo "$CMD"

echo ""
echo "========================================"
echo "SECTION 6: ccache debug replay"
echo "========================================"
cd "$DIR"
CCACHE_DEBUG=1 CCACHE_DEBUGDIR="$TMPDIR" \
  CCACHE_LOGFILE="$TMPDIR/ccache.log" \
  ccache $CMD -o /dev/null 2>&1 || true

echo "-- log --"
cat "$TMPDIR/ccache.log" 2>/dev/null || true
echo "-- debug --"
grep -r "hash\|key\|manifest\|Result\|base_dir\|compiler" "$TMPDIR/"*.ccache-debug 2>/dev/null || true

echo ""
echo "========================================"
echo "SECTION 7: preprocessed output (.ii)"
echo "========================================"
# Strip the -o flag and replace the compiler with clang++ -E to preprocess only.
# Also strip -c so clang doesn't complain about -E -c together.
PREPROCESS_CMD=$(echo "$CMD" \
  | sed 's| -o [^ ]*||g' \
  | sed 's| -c | |g')

# Extract the compiler (first token) and the rest of the flags
COMPILER=$(echo "$PREPROCESS_CMD" | awk '{print $1}')
FLAGS=$(echo "$PREPROCESS_CMD" | cut -d' ' -f2-)

$COMPILER -E $FLAGS -o /dev/stdout 2>/dev/null
