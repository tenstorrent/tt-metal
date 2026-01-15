#!/bin/bash
# Rename generic operation types to operation-specific names, build, commit if ok
# Usage: ./run_operation_type_naming_fix.sh [--dry-run] [--check-only] [path-pattern]

PLUGIN=/home/boxx/tt-clang-tidy-checks/build/ttnn-operation-type-naming/TtNNOperationTypeNamingCheck.so
TT_METAL_ROOT=/home/boxx/tt-metal
BUILD_DIR="$TT_METAL_ROOT/build"
PROGRESS_FILE="$TT_METAL_ROOT/.operation_type_naming_progress"
EXCEPTIONS_FILE="$TT_METAL_ROOT/.operation_type_naming_exceptions"

cd "$TT_METAL_ROOT"

# Check prerequisites
if [ ! -f "$PLUGIN" ]; then
  echo "ERROR: Plugin not found at $PLUGIN"
  echo "Build it: cd /home/boxx/tt-clang-tidy-checks/build && make"
  exit 1
fi

if [ ! -f "$BUILD_DIR/compile_commands.json" ]; then
  echo "ERROR: compile_commands.json not found at $BUILD_DIR"
  exit 1
fi

DRY_RUN=0
CHECK_ONLY=0
PATH_FILTER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    *)
      PATH_FILTER="$1"
      shift
      ;;
  esac
done

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
if [ -n "$PATH_FILTER" ]; then
  DIRS=$(find "$TT_METAL_ROOT/ttnn/cpp/ttnn/operations" -path "*$PATH_FILTER*" -name "*_device_operation_types.hpp" -exec dirname {} \; 2>/dev/null | sort -u)
else
  DIRS=$(find "$TT_METAL_ROOT/ttnn/cpp/ttnn/operations" -name "*_device_operation_types.hpp" -exec dirname {} \; 2>/dev/null | sort -u)
fi

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
    -checks='-*,ttnn-operation-type-naming' \
    -p "$BUILD_DIR" \
    $FILES 2>&1) || true

  if ! echo "$CHECK_OUTPUT" | grep -q "warning:"; then
    echo "  No changes needed"
    echo "$DEVICE_DIR" >> "$PROGRESS_FILE"
    NO_CHANGE_COUNT=$((NO_CHANGE_COUNT + 1))
    continue
  fi

  # Show what would be changed
  WARNINGS=$(echo "$CHECK_OUTPUT" | grep "warning:" | head -5)
  echo "  Issues found:"
  echo "$WARNINGS" | while read -r line; do
    echo "    $line"
  done

  if [ "$CHECK_ONLY" -eq 1 ]; then
    echo "  (check-only mode, not fixing)"
    continue
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  (dry-run mode, not fixing)"
    continue
  fi

  # Apply fixes
  echo "  Applying fixes..."
  FIX_OUTPUT=$(clang-tidy-17 -load "$PLUGIN" \
    -checks='-*,ttnn-operation-type-naming' \
    -fix-errors \
    -p "$BUILD_DIR" \
    $FILES 2>&1) || true

  APPLIED=$(echo "$FIX_OUTPUT" | grep -c "applied" || echo 0)
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
refactor($OP_NAME): rename operation types to ${OP_NAME^}Params/${OP_NAME^}Inputs

Rename generic type names to operation-specific names:
- operation_attributes_t -> ${OP_NAME^}Params
- tensor_args_t -> ${OP_NAME^}Inputs

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
echo "  No changes needed: $NO_CHANGE_COUNT"
echo "  Skipped (already done): $SKIPPED_COUNT"
echo "  Added to exceptions: $EXCEPTION_COUNT"
echo "========================================"
