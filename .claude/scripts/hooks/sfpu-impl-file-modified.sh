#!/bin/bash
# PostToolUse hook for sfpu-operation-implementor: remind agent to track file in implementation notes
INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool_name // empty')

[[ "$TOOL" == "Write" || "$TOOL" == "Edit" ]] || exit 0

FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

case "$FILE_PATH" in
  *ckernel_sfpu_*.h|*llk_math_eltwise_unary_sfpu_*.h)
    jq -n --arg file "$FILE_PATH" '{
      hookSpecificOutput: {
        hookEventName: "PostToolUse",
        additionalContext: ("You just modified an SFPU kernel file: " + $file + ". Ensure this file is listed in ### New Files (if created) or ### Modified Files (if edited) in your implementation notes.")
      }
    }'
    ;;
  *eltwise_unary/*.h|*sfpu_split_includes.h|*llk_sfpu_types.h)
    jq -n --arg file "$FILE_PATH" '{
      hookSpecificOutput: {
        hookEventName: "PostToolUse",
        additionalContext: ("You just modified a compute API / include file: " + $file + ". Ensure this file is listed in ### New Files or ### Modified Files in your implementation notes.")
      }
    }'
    ;;
  *unary_op_types.hpp|*unary_op_utils.cpp|*unary_op_utils.hpp|*unary.hpp|*unary.cpp|*unary_nanobind.cpp|*unary.py)
    jq -n --arg file "$FILE_PATH" '{
      hookSpecificOutput: {
        hookEventName: "PostToolUse",
        additionalContext: ("You just modified a registration file: " + $file + ". Ensure this file is listed in ### Modified Files in your implementation notes.")
      }
    }'
    ;;
  *)
    exit 0
    ;;
esac
