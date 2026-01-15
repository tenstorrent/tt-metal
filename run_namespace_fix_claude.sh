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
echo "Already processed: ${#PROCESSED[@]}"
echo "Exceptions: ${#EXCEPTIONS[@]}"
echo "Remaining to process: $((TOTAL_DIRS - ${#PROCESSED[@]} - ${#EXCEPTIONS[@]}))"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

FIXED_COUNT=0
SKIPPED_COUNT=0
EXCEPTION_COUNT=0
CURRENT_NUM=0
START_TIME=$(date +%s)

for DEVICE_DIR in $DIRS; do
  CURRENT_NUM=$((CURRENT_NUM + 1))
  OP_NAME=$(basename "$(dirname "$DEVICE_DIR")")
  PARENT_DIR=$(dirname "$DEVICE_DIR")
  OP_START_TIME=$(date +%s)

  # Skip already processed
  if [ "${PROCESSED[$DEVICE_DIR]}" == "1" ]; then
    echo "[$CURRENT_NUM/$TOTAL_DIRS] SKIP (done): $OP_NAME"
    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    continue
  fi

  # Skip exceptions
  if [ "${EXCEPTIONS[$DEVICE_DIR]}" == "1" ]; then
    echo "[$CURRENT_NUM/$TOTAL_DIRS] SKIP (exception): $OP_NAME"
    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    continue
  fi

  echo ""
  echo "============================================"
  echo "[$CURRENT_NUM/$TOTAL_DIRS] Processing: $OP_NAME"
  echo "  Directory: $DEVICE_DIR"
  echo "  Started at: $(date '+%Y-%m-%d %H:%M:%S')"

  # Get all source files in device dir and parent dir
  echo "  [Step 1/5] Finding source files..."
  FILES=$(find "$DEVICE_DIR" "$PARENT_DIR" -maxdepth 1 \( -name "*.hpp" -o -name "*.cpp" \) 2>/dev/null | sort -u)

  if [ -z "$FILES" ]; then
    echo "  SKIP: no source files found"
    continue
  fi

  FILE_COUNT=$(echo "$FILES" | wc -l)
  echo "  [Step 1/5] Found $FILE_COUNT source files:"
  echo "$FILES" | sed 's/^/    - /'

  # Check if already in ttnn::prim namespace
  echo "  [Step 2/5] Checking if namespace changes are needed..."
  if ! grep -l "ttnn::operations::" $FILES >/dev/null 2>&1; then
    echo "  [Step 2/5] No changes needed (already migrated or no matching patterns)"
    echo "$DEVICE_DIR" >> "$PROGRESS_FILE"
    OP_ELAPSED=$(($(date +%s) - OP_START_TIME))
    echo "  Completed in ${OP_ELAPSED}s"
    continue
  fi

  # Determine target namespace
  echo "  [Step 3/5] Determining target namespace..."
  TARGET_NS="ttnn::prim"
  if echo "$DEVICE_DIR" | grep -q "/operations/experimental/"; then
    TARGET_NS="ttnn::experimental::prim"
  fi
  echo "  [Step 3/5] Target namespace: $TARGET_NS"

  # Create prompt for Claude
  echo "  [Step 4/5] Preparing Claude prompt..."
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

  echo "  [Step 4/5] Launching Claude Code..."
  echo "  [Step 4/5] This may take several minutes..."
  CLAUDE_START=$(date +%s)
  LOG_FILE="/tmp/claude_${OP_NAME}.log"

  # Initialize log file
  > "$LOG_FILE"

  # Start background monitor for progress updates
  (
    LAST_LOG_SIZE=0
    LAST_GIT_STATUS=""
    LAST_LOG_TIME=0
    UPDATE_COUNT=0
    STALE_COUNT=0

    while true; do
      sleep 5  # Update every 5 seconds
      UPDATE_COUNT=$((UPDATE_COUNT + 1))
      ELAPSED=$(($(date +%s) - CLAUDE_START))

      # Check if log file is still being written (Claude still running)
      if [ -f "$LOG_FILE" ]; then
        CURRENT_LOG_SIZE=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
        CURRENT_LOG_TIME=$(stat -f%m "$LOG_FILE" 2>/dev/null || stat -c%Y "$LOG_FILE" 2>/dev/null || echo 0)
        LOG_GROWTH=$((CURRENT_LOG_SIZE - LAST_LOG_SIZE))

        # Check if log file has been updated recently
        TIME_SINCE_LOG_UPDATE=$(($(date +%s) - CURRENT_LOG_TIME))
        if [ $CURRENT_LOG_TIME -eq $LAST_LOG_TIME ] && [ $LAST_LOG_TIME -gt 0 ]; then
          STALE_COUNT=$((STALE_COUNT + 1))
          # If log hasn't been updated in 30+ seconds, assume Claude is done
          if [ $TIME_SINCE_LOG_UPDATE -gt 30 ]; then
            break
          fi
        else
          STALE_COUNT=0
        fi

        LAST_LOG_SIZE=$CURRENT_LOG_SIZE
        LAST_LOG_TIME=$CURRENT_LOG_TIME

        # Get last meaningful line of log for context
        LAST_LINE=$(tail -n 5 "$LOG_FILE" 2>/dev/null | grep -v "^$" | grep -v "^[[:space:]]*$" | tail -n 1 | sed 's/^[[:space:]]*//' | cut -c1-80)
      else
        LOG_GROWTH=0
        LAST_LINE=""
        STALE_COUNT=$((STALE_COUNT + 1))
        if [ $STALE_COUNT -gt 6 ]; then
          break
        fi
      fi

      # Check git status for file changes
      CURRENT_GIT_STATUS=$(git status --porcelain ttnn/ 2>/dev/null | head -n 3 | wc -l)
      GIT_CHANGED=""
      if [ "$CURRENT_GIT_STATUS" != "$LAST_GIT_STATUS" ] && [ "$CURRENT_GIT_STATUS" -gt 0 ]; then
        GIT_CHANGED=" (${CURRENT_GIT_STATUS} file(s) modified)"
        LAST_GIT_STATUS="$CURRENT_GIT_STATUS"
      fi

      # Print status update (always show every 30s, or when something changes)
      if [ $UPDATE_COUNT -eq 1 ] || [ $LOG_GROWTH -gt 100 ] || [ -n "$GIT_CHANGED" ] || [ $((UPDATE_COUNT % 6)) -eq 0 ]; then
        STATUS_MSG="  [Step 4/5] Running... (${ELAPSED}s elapsed"
        if [ $LOG_GROWTH -gt 0 ]; then
          STATUS_MSG="${STATUS_MSG}, +${LOG_GROWTH} bytes"
        fi
        if [ -n "$GIT_CHANGED" ]; then
          STATUS_MSG="${STATUS_MSG}${GIT_CHANGED}"
        fi
        STATUS_MSG="${STATUS_MSG})"
        echo "$STATUS_MSG" >&2

        # Show last log line if available and meaningful
        if [ -n "$LAST_LINE" ] && [ ${#LAST_LINE} -gt 5 ]; then
          echo "    → ${LAST_LINE}..." >&2
        fi
      fi
    done
  ) &
  MONITOR_PID=$!

  # Run Claude Code with Sonnet (--dangerously-skip-permissions for batch mode)
  claude -p "$PROMPT" --model sonnet --dangerously-skip-permissions 2>&1 | tee "$LOG_FILE"

  CLAUDE_EXIT=$?

  # Stop the monitor
  kill $MONITOR_PID 2>/dev/null
  wait $MONITOR_PID 2>/dev/null

  CLAUDE_ELAPSED=$(($(date +%s) - CLAUDE_START))
  echo "  [Step 4/5] Claude execution completed (exit code: $CLAUDE_EXIT, elapsed: ${CLAUDE_ELAPSED}s)"

  # Check if changes were committed
  echo "  [Step 5/5] Checking results..."
  LAST_COMMIT=$(git log -1 --format="%s" 2>/dev/null)
  if echo "$LAST_COMMIT" | grep -q "refactor(namespace): simplify $OP_NAME"; then
    echo "  [Step 5/5] ✓ SUCCESS: Changes committed"
    echo "$DEVICE_DIR" >> "$PROGRESS_FILE"
    FIXED_COUNT=$((FIXED_COUNT + 1))
  else
    # Check if there are uncommitted changes
    if ! git diff --quiet ttnn/ 2>/dev/null; then
      echo "  [Step 5/5] ✗ FAILED: Uncommitted changes detected, reverting..."
      git checkout -- ttnn/
      echo "  [Step 5/5] Changes reverted"
    else
      echo "  [Step 5/5] ✗ FAILED: No commit found and no uncommitted changes"
    fi
    echo "  [Step 5/5] Adding to exceptions list"
    echo "$DEVICE_DIR" >> "$EXCEPTIONS_FILE"
    EXCEPTION_COUNT=$((EXCEPTION_COUNT + 1))
  fi

  OP_ELAPSED=$(($(date +%s) - OP_START_TIME))
  TOTAL_ELAPSED=$(($(date +%s) - START_TIME))
  REMAINING=$((TOTAL_DIRS - CURRENT_NUM))

  echo "  Operation completed in ${OP_ELAPSED}s"
  echo "  Progress: $FIXED_COUNT fixed, $EXCEPTION_COUNT exceptions, $SKIPPED_COUNT skipped"
  echo "  Total elapsed: ${TOTAL_ELAPSED}s (~$((TOTAL_ELAPSED / 60))m)"
  if [ $REMAINING -gt 0 ] && [ $CURRENT_NUM -gt 0 ]; then
    AVG_TIME=$((TOTAL_ELAPSED / CURRENT_NUM))
    ESTIMATED_REMAINING=$((AVG_TIME * REMAINING))
    echo "  Estimated time remaining: ~${ESTIMATED_REMAINING}s (~$((ESTIMATED_REMAINING / 60))m)"
  fi
  echo ""
done

TOTAL_ELAPSED=$(($(date +%s) - START_TIME))
echo ""
echo "========================================"
echo "Final Summary:"
echo "  Fixed and committed: $FIXED_COUNT"
echo "  Skipped (done/exc):  $SKIPPED_COUNT"
echo "  Added to exceptions: $EXCEPTION_COUNT"
echo "  Total operations:    $TOTAL_DIRS"
echo "  Total time:          ${TOTAL_ELAPSED}s (~$((TOTAL_ELAPSED / 60))m)"
if [ $CURRENT_NUM -gt 0 ]; then
  AVG_TIME=$((TOTAL_ELAPSED / CURRENT_NUM))
  echo "  Average per operation: ${AVG_TIME}s"
fi
echo "  Completed at:        $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
