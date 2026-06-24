#!/usr/bin/env bash
# ccache-diagnose.sh — run on each agent, compare outputs
set -euo pipefail

BUILD_DIR="${1:?Usage: $0 <cmake-build-dir>}"

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Pick the first .cpp entry from compile_commands.json
ENTRY=$(jq 'map(select(.file | test("\\.cpp$|\\.cc$|\\.cxx$"))) | first' \
  "$BUILD_DIR/compile_commands.json")
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
