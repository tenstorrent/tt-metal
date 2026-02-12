#!/usr/bin/env bash
# Run clang-tidy on changed C/C++ files before commit.
# Requires: build/compile_commands.json (run cmake with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON).
# Usage (from repo root):
#   scripts/run_clang_tidy_before_commit.sh
#     → runs on staged or changed C/C++ files that are in compile_commands.json
#   scripts/run_clang_tidy_before_commit.sh file1.cpp file2.hpp ...
#     → runs on the given file(s) that are in compile_commands.json
set -e
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
BUILD_DIR="${BUILD_DIR:-build}"
COMPILE_DB="$BUILD_DIR/compile_commands.json"

if [[ ! -f "$COMPILE_DB" ]]; then
  echo "clang-tidy: $COMPILE_DB not found (no build or compile_commands). Skipping."
  echo "  To enable: cmake -B $BUILD_DIR -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ... && cmake --build $BUILD_DIR --target all_generated_files"
  exit 0
fi

CLANG_TIDY=""
for cmd in clang-tidy-20 clang-tidy-17 clang-tidy; do
  if command -v "$cmd" >/dev/null 2>&1; then
    CLANG_TIDY="$cmd"
    break
  fi
done
if [[ -z "$CLANG_TIDY" ]]; then
  echo "clang-tidy: no clang-tidy found. Install e.g. clang-tidy-20. Skipping."
  exit 0
fi

# Files to check: from args, or staged, or changed under repo
if [[ $# -gt 0 ]]; then
  FILES="$*"
else
  FILES="$(git diff --cached --name-only --diff-filter=ACM 2>/dev/null | grep -E '\.(cpp|cc|cxx|c|hpp|h|hxx)$' || true)"
  if [[ -z "$FILES" ]]; then
    FILES="$(git diff --name-only HEAD 2>/dev/null | grep -E '\.(cpp|cc|cxx|c|hpp|h|hxx)$' || true)"
  fi
fi
if [[ -z "$FILES" ]]; then
  echo "clang-tidy: no C/C++ files to check."
  exit 0
fi

# Filter to files present in compile_commands.json (relative paths from repo root)
if command -v jq >/dev/null 2>&1; then
  PREFIX="${ROOT}/"
  IN_BUILD="$(mktemp)"
  jq --arg prefix "$PREFIX" -r '.[].file | sub("^" + $prefix; "")' "$COMPILE_DB" > "$IN_BUILD"
  TO_CHECK=""
  for f in $FILES; do
    if grep -q -F "$f" "$IN_BUILD" 2>/dev/null; then
      TO_CHECK="$TO_CHECK $f"
    fi
  done
  rm -f "$IN_BUILD"
  FILES="$(echo "$TO_CHECK" | xargs)"
fi

if [[ -z "$FILES" ]]; then
  echo "clang-tidy: no changed files are in compile_commands.json. Skipping."
  exit 0
fi

echo "clang-tidy: checking: $FILES"
FAILED=0
for f in $FILES; do
  if [[ -f "$f" ]]; then
    if ! "$CLANG_TIDY" -p "$BUILD_DIR" -quiet "$f" 2>/dev/null; then
      FAILED=1
    fi
  fi
done
exit $FAILED
