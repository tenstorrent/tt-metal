#!/bin/bash
# Run namespace simplification using Claude Code with Sonnet
#
# For each device operation directory:
# 1. Launch Claude Code to fix namespaces
# 2. Build to verify
# 3. Commit on success, add to exceptions on failure

TT_METAL_ROOT=/home/boxx/tt-metal
PROGRESS_FILE="$TT_METAL_ROOT/.namespace_claude_progress"
EXCEPTIONS_FILE="$TT_METAL_ROOT/.namespace_claude_exceptions"

cd "$TT_METAL_ROOT"

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
echo "Found $TOTAL_DIRS operations with device operation types files"
echo ""

FIXED_COUNT=0
SKIPPED_COUNT=0
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

  echo "============================================"
  echo "Processing: $OP_NAME"
  echo "  Directory: $DEVICE_DIR"

  # Get all source files in device dir and parent dir
  FILES=$(find "$DEVICE_DIR" "$PARENT_DIR" -maxdepth 1 \( -name "*.hpp" -o -name "*.cpp" \) 2>/dev/null | sort -u)

  if [ -z "$FILES" ]; then
    echo "  SKIP: no source files found"
    continue
  fi

  FILE_COUNT=$(echo "$FILES" | wc -l)
  echo "  Found $FILE_COUNT source files"

  # Check if already in ttnn::prim namespace
  if ! grep -l "ttnn::operations::" $FILES >/dev/null 2>&1; then
    echo "  No changes needed (already migrated or no matching patterns)"
    echo "$DEVICE_DIR" >> "$PROGRESS_FILE"
    continue
  fi

  # Determine target namespace
  TARGET_NS="ttnn::prim"
  if echo "$DEVICE_DIR" | grep -q "/operations/experimental/"; then
    TARGET_NS="ttnn::experimental::prim"
  fi

  # Create prompt for Claude
  PROMPT="Fix the namespace in these device operation files to use $TARGET_NS.

Files to modify (in $DEVICE_DIR and $PARENT_DIR):
$FILES

Rules:
1. Change namespace declarations from ttnn::operations::*::* to $TARGET_NS
2. Change ttnn::operations::*::*::program namespace to $TARGET_NS
3. Update closing namespace comments
4. Replace relative qualifiers like '${OP_NAME}::SomeType' with '$TARGET_NS::SomeType'
5. Replace 'program::SomeFactory' with '$TARGET_NS::SomeFactory' or just 'SomeFactory' if in same namespace
6. Update any fully-qualified references ttnn::operations::*::*::Type to $TARGET_NS::Type
7. Do NOT modify files outside the device/ directory (like nanobind files, public API files)
8. Keep the struct/class names the same, only change namespaces

After making changes, verify the code compiles by running: ./build_metal.sh -c -e --debug --build-all

If build succeeds, commit with message: refactor(namespace): simplify $OP_NAME to $TARGET_NS
If build fails, revert changes with: git checkout -- ttnn/"

  echo "  Launching Claude Code..."

  # Run Claude Code with Sonnet (--dangerously-skip-permissions for batch mode)
  claude -p "$PROMPT" --model sonnet --dangerously-skip-permissions 2>&1 | tee "/tmp/claude_${OP_NAME}.log"

  CLAUDE_EXIT=$?

  # Check if changes were committed
  LAST_COMMIT=$(git log -1 --format="%s" 2>/dev/null)
  if echo "$LAST_COMMIT" | grep -q "refactor(namespace): simplify $OP_NAME"; then
    echo "  SUCCESS: Changes committed"
    echo "$DEVICE_DIR" >> "$PROGRESS_FILE"
    FIXED_COUNT=$((FIXED_COUNT + 1))
  else
    # Check if there are uncommitted changes
    if ! git diff --quiet ttnn/ 2>/dev/null; then
      echo "  FAILED: Uncommitted changes, reverting..."
      git checkout -- ttnn/
    fi
    echo "  Adding to exceptions"
    echo "$DEVICE_DIR" >> "$EXCEPTIONS_FILE"
    EXCEPTION_COUNT=$((EXCEPTION_COUNT + 1))
  fi

  echo ""
done

echo "========================================"
echo "Summary:"
echo "  Fixed and committed: $FIXED_COUNT"
echo "  Skipped (done/exc):  $SKIPPED_COUNT"
echo "  Added to exceptions: $EXCEPTION_COUNT"
echo "========================================"
