#!/bin/bash
# Run clang-tidy fix one file at a time, build, commit if ok, skip if not

PLUGIN=/home/boxx/tt-clang-tidy-checks/build/ttnn-nanobind-overload/TtNNNanobindOverloadCheck.so
CC_JSON=/home/boxx/tt-metal/build_Debug/compile_commands.json
TT_METAL_ROOT=/home/boxx/tt-metal
PROGRESS_FILE="$TT_METAL_ROOT/.nanobind_fix_progress"
EXCEPTIONS_FILE="$TT_METAL_ROOT/.nanobind_fix_exceptions"

cd "$TT_METAL_ROOT"

# Check prerequisites
if [ ! -f "$PLUGIN" ]; then
  echo "ERROR: Plugin not found at $PLUGIN"
  exit 1
fi

if [ ! -f "$CC_JSON" ]; then
  echo "ERROR: compile_commands.json not found"
  exit 1
fi

# Load already processed files
declare -A PROCESSED
if [ -f "$PROGRESS_FILE" ]; then
  while IFS= read -r line; do
    PROCESSED["$line"]=1
  done < "$PROGRESS_FILE"
  echo "Loaded ${#PROCESSED[@]} already processed files"
fi

# Load exceptions
declare -A EXCEPTIONS
if [ -f "$EXCEPTIONS_FILE" ]; then
  while IFS= read -r line; do
    EXCEPTIONS["$line"]=1
  done < "$EXCEPTIONS_FILE"
  echo "Loaded ${#EXCEPTIONS[@]} exception files"
fi

# Find all nanobind files in ttnn only
FILES=$(find "$TT_METAL_ROOT/ttnn" -name "*nanobind*.cpp" -type f 2>/dev/null | sort)

TOTAL_FILES=$(echo "$FILES" | wc -l)
echo "Found $TOTAL_FILES nanobind files in ttnn/"
echo ""

FIXED_COUNT=0
SKIPPED_COUNT=0
EXCEPTION_COUNT=0

for FILE in $FILES; do
  # Skip already processed files
  if [ "${PROCESSED[$FILE]}" == "1" ]; then
    echo "SKIP (done): $FILE"
    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    continue
  fi

  # Skip exception files
  if [ "${EXCEPTIONS[$FILE]}" == "1" ]; then
    echo "SKIP (exception): $FILE"
    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    continue
  fi

  echo "Processing: $FILE"

  # Extract compile flags
  FLAGS=$(python3 << EOF
import json
with open("$CC_JSON") as f:
    data = json.load(f)
for entry in data:
    if "$FILE" in entry.get('file', ''):
        cmd = entry['command']
        parts = cmd.split()
        filtered = []
        skip_next = False
        for p in parts[1:]:
            if skip_next:
                skip_next = False
                continue
            if p == '-c' or p == '-o':
                skip_next = (p == '-o')
                continue
            if p.endswith('.cpp'):
                continue
            filtered.append(p)
        print(' '.join(filtered))
        break
EOF
)

  if [ -z "$FLAGS" ]; then
    echo "  SKIP: not in compile_commands.json"
    continue
  fi

  # Run clang-tidy with fix
  OUTPUT=$(clang-tidy-17 -load "$PLUGIN" \
    -checks='-*,ttnn-nanobind-unnecessary-overload' \
    --fix \
    "$FILE" \
    -- $FLAGS 2>&1) || true

  # Check if any fix was applied
  if ! echo "$OUTPUT" | grep -q "unnecessary use of nanobind_overload_t"; then
    echo "  No changes needed"
    echo "$FILE" >> "$PROGRESS_FILE"
    continue
  fi

  echo "  Fix applied, building..."

  # Build to verify
  BUILD_OUTPUT=$(./build_metal.sh -c -e --debug --build-all 2>&1)
  BUILD_STATUS=$?

  if [ $BUILD_STATUS -eq 0 ]; then
    echo "  Build OK, committing..."

    # Only stage changes in ttnn/
    git add ttnn/

    # Check if there are staged changes
    if git diff --cached --quiet; then
      echo "  No changes to commit"
      echo "$FILE" >> "$PROGRESS_FILE"
    else
      git commit -m "$(cat <<EOF
refactor(nanobind): simplify $(basename $FILE .cpp)

Replace nanobind_overload_t with nanobind_arguments_t (single overload).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
      echo "  Committed!"
      echo "$FILE" >> "$PROGRESS_FILE"
      FIXED_COUNT=$((FIXED_COUNT + 1))
    fi
  else
    echo "  BUILD FAILED - adding to exceptions"

    # Revert changes to this file
    git checkout -- "$FILE"

    # Also revert any related files that might have been changed
    git checkout -- ttnn/ 2>/dev/null || true

    # Add to exceptions
    echo "$FILE" >> "$EXCEPTIONS_FILE"
    EXCEPTION_COUNT=$((EXCEPTION_COUNT + 1))
  fi

  echo ""
done

echo "========================================"
echo "Summary:"
echo "  Fixed and committed: $FIXED_COUNT"
echo "  Skipped (already done): $SKIPPED_COUNT"
echo "  Added to exceptions: $EXCEPTION_COUNT"
echo "========================================"
