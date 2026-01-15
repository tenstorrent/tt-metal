#!/bin/bash
# Run namespace simplification clang-tidy fix by operation directory, build, commit if ok, skip if not
#
# This script processes device operation directories and simplifies namespace patterns:
#   ttnn::operations::<category>::* -> ttnn::prim
#   ttnn::operations::experimental::* -> ttnn::experimental::prim
#
# Based on run_return_value_alias_fix.sh - processes all files in an operation together

PLUGIN=/home/boxx/tt-clang-tidy-checks/build/ttnn-namespace-simplification/TtNNNamespaceSimplificationCheck.so
TT_METAL_ROOT=/home/boxx/tt-metal
BUILD_DIR="$TT_METAL_ROOT/build_Debug"
PROGRESS_FILE="$TT_METAL_ROOT/.namespace_fix_progress"
EXCEPTIONS_FILE="$TT_METAL_ROOT/.namespace_fix_exceptions"

cd "$TT_METAL_ROOT"

# Check prerequisites
if [ ! -f "$PLUGIN" ]; then
  echo "ERROR: Plugin not found at $PLUGIN"
  echo "Build it with:"
  echo "  cd /home/boxx/tt-clang-tidy-checks"
  echo "  mkdir -p build && cd build"
  echo "  cmake .. -DCLANG_VERSION=17"
  echo "  make -j\$(nproc)"
  exit 1
fi

if [ ! -f "$BUILD_DIR/compile_commands.json" ]; then
  echo "ERROR: compile_commands.json not found at $BUILD_DIR"
  echo "Build tt-metal first to generate compile_commands.json"
  exit 1
fi

# Load already processed operations
declare -A PROCESSED
if [ -f "$PROGRESS_FILE" ]; then
  while IFS= read -r line; do
    PROCESSED["$line"]=1
  done < "$PROGRESS_FILE"
  echo "Loaded ${#PROCESSED[@]} already processed operations"
fi

# Load exceptions
declare -A EXCEPTIONS
if [ -f "$EXCEPTIONS_FILE" ]; then
  while IFS= read -r line; do
    EXCEPTIONS["$line"]=1
  done < "$EXCEPTIONS_FILE"
  echo "Loaded ${#EXCEPTIONS[@]} exception operations"
fi

# Find all device directories with types files
DIRS=$(find "$TT_METAL_ROOT/ttnn/cpp/ttnn/operations" -name "*_device_operation_types.hpp" -exec dirname {} \; 2>/dev/null | sort -u)

TOTAL_DIRS=$(echo "$DIRS" | grep -c . || echo 0)
echo "Found $TOTAL_DIRS operations with types files"
echo ""

FIXED_COUNT=0
SKIPPED_COUNT=0
NO_CHANGE_COUNT=0
EXCEPTION_COUNT=0

for DEVICE_DIR in $DIRS; do
  OP_NAME=$(basename "$(dirname "$DEVICE_DIR")")
  PARENT_DIR=$(dirname "$DEVICE_DIR")

  # Skip already processed
  if [ "${PROCESSED[$DEVICE_DIR]}" == "1" ]; then
    echo "SKIP (done): $OP_NAME"
    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    continue
  fi

  # Skip exceptions
  if [ "${EXCEPTIONS[$DEVICE_DIR]}" == "1" ]; then
    echo "SKIP (exception): $OP_NAME"
    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    continue
  fi

  echo "Processing: $OP_NAME"
  echo "  Directory: $DEVICE_DIR"

  # Get all source files in device dir and parent dir
  FILES=$(find "$DEVICE_DIR" "$PARENT_DIR" -maxdepth 1 \( -name "*.hpp" -o -name "*.cpp" \) 2>/dev/null | sort -u | tr '\n' ' ')

  if [ -z "$FILES" ]; then
    echo "  SKIP: no source files found"
    continue
  fi

  FILE_COUNT=$(echo $FILES | wc -w)
  echo "  Found $FILE_COUNT source files"

  # Check if there are any issues to fix
  CHECK_OUTPUT=$(clang-tidy-17 -load "$PLUGIN" \
    -checks='-*,ttnn-namespace-simplification' \
    -p "$BUILD_DIR" \
    $FILES 2>&1) || true

  if ! echo "$CHECK_OUTPUT" | grep -q "warning:"; then
    echo "  No changes needed"
    echo "$DEVICE_DIR" >> "$PROGRESS_FILE"
    NO_CHANGE_COUNT=$((NO_CHANGE_COUNT + 1))
    continue
  fi

  # Apply fixes
  echo "  Applying fixes..."
  FIX_OUTPUT=$(clang-tidy-17 -load "$PLUGIN" \
    -checks='-*,ttnn-namespace-simplification' \
    --fix-errors \
    -p "$BUILD_DIR" \
    $FILES 2>&1) || true

  APPLIED=$(echo "$FIX_OUTPUT" | grep -c "FIX-IT applied" || echo 0)
  echo "  Applied $APPLIED fixes"

  if [ "$APPLIED" -eq 0 ]; then
    echo "  No fixes applied"
    echo "$DEVICE_DIR" >> "$PROGRESS_FILE"
    NO_CHANGE_COUNT=$((NO_CHANGE_COUNT + 1))
    continue
  fi

  # Build to verify
  echo "  Building..."
  BUILD_OUTPUT=$(./build_metal.sh -c -e --debug --build-all 2>&1)
  BUILD_STATUS=$?

  if [ $BUILD_STATUS -eq 0 ]; then
    echo "  Build OK, committing..."

    # Stage changes in ttnn/
    git add ttnn/

    # Check if there are staged changes
    if git diff --cached --quiet; then
      echo "  No changes to commit"
      echo "$DEVICE_DIR" >> "$PROGRESS_FILE"
    else
      git commit -m "$(cat <<EOF
refactor(namespace): simplify $OP_NAME to ttnn::prim

Migrate from ttnn::operations::<category> to ttnn::prim namespace.

See: docs/NAMESPACE_SIMPLIFICATION_GUIDE.md

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
      echo "  Committed!"
      echo "$DEVICE_DIR" >> "$PROGRESS_FILE"
      FIXED_COUNT=$((FIXED_COUNT + 1))
    fi
  else
    echo "  BUILD FAILED - adding to exceptions"

    # Revert changes
    git checkout -- ttnn/ 2>/dev/null || true

    # Add to exceptions
    echo "$DEVICE_DIR" >> "$EXCEPTIONS_FILE"
    EXCEPTION_COUNT=$((EXCEPTION_COUNT + 1))
  fi

  echo ""
done

echo "========================================"
echo "Summary:"
echo "  Fixed and committed: $FIXED_COUNT"
echo "  No changes needed:   $NO_CHANGE_COUNT"
echo "  Skipped (done/exc):  $SKIPPED_COUNT"
echo "  Added to exceptions: $EXCEPTION_COUNT"
echo "========================================"
