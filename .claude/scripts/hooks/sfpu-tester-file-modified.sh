#!/bin/bash
# PostToolUse hook for sfpu-operation-tester: remind agent to log fix and track file
INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool_name // empty')

[[ "$TOOL" == "Write" || "$TOOL" == "Edit" ]] || exit 0

FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

case "$FILE_PATH" in
  *test_*.py)
    # Test file creation — no fix_applied needed
    jq -n --arg file "$FILE_PATH" '{
      hookSpecificOutput: {
        hookEventName: "PostToolUse",
        additionalContext: ("You just created/modified a test file: " + $file + ". Log a test_created breadcrumb if this is the initial creation. Ensure this file is listed in ### New Files in the implementation notes.")
      }
    }'
    ;;
  *ckernel_sfpu_*.h|*llk_math_eltwise_unary_sfpu_*.h)
    jq -n --arg file "$FILE_PATH" '{
      hookSpecificOutput: {
        hookEventName: "PostToolUse",
        additionalContext: ("You just modified an SFPU kernel file: " + $file + " as a debugging fix. You MUST: 1) Log a fix_applied breadcrumb with the change description. 2) Ensure BOTH architectures (wormhole_b0 and blackhole) are updated identically. 3) Update ### Modified Files in the implementation notes.")
      }
    }'
    ;;
  *eltwise_unary/*.h|*sfpu_split_includes.h|*llk_sfpu_types.h)
    jq -n --arg file "$FILE_PATH" '{
      hookSpecificOutput: {
        hookEventName: "PostToolUse",
        additionalContext: ("You just modified a compute API / include file: " + $file + " as a debugging fix. You MUST: 1) Log a fix_applied breadcrumb. 2) Update ### Modified Files in the implementation notes.")
      }
    }'
    ;;
  *unary_op_types.hpp|*unary_op_utils.cpp|*unary_op_utils.hpp|*unary.hpp|*unary.cpp|*unary_nanobind.cpp|*unary.py)
    jq -n --arg file "$FILE_PATH" '{
      hookSpecificOutput: {
        hookEventName: "PostToolUse",
        additionalContext: ("You just modified a registration file: " + $file + " as a debugging fix. You MUST: 1) Log a fix_applied breadcrumb. 2) Update ### Modified Files in the implementation notes.")
      }
    }'
    ;;
  *)
    exit 0
    ;;
esac
