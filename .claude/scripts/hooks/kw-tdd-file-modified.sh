#!/bin/bash
# PostToolUse hook for kernel-writer-tdd: remind agent to log breadcrumbs after file modifications
INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool_name // empty')

# Only trigger on Write or Edit tools
[[ "$TOOL" == "Write" || "$TOOL" == "Edit" ]] || exit 0

FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Only trigger for kernel or python operation files
case "$FILE_PATH" in
  *kernels/*.cpp)
    jq -n --arg file "$FILE_PATH" '{
      hookSpecificOutput: {
        hookEventName: "PostToolUse",
        additionalContext: ("You just modified a kernel file: " + $file + ". Log a kernel_implemented breadcrumb NOW:\n.claude/scripts/logging/append_breadcrumb.sh \"{op_path}\" \"ttnn-kernel-writer-tdd\" '"'"'{\"event\":\"kernel_implemented\",\"stage\":\"CURRENT_STAGE\",\"kernel\":\"KERNEL_NAME\",\"approach\":\"WHAT_YOU_CHANGED\"}'"'"'")
      }
    }'
    ;;
  *_program_descriptor.py|*layer_norm*.py|*__init__.py)
    jq -n --arg file "$FILE_PATH" '{
      hookSpecificOutput: {
        hookEventName: "PostToolUse",
        additionalContext: ("You just modified an upstream file: " + $file + ". Log an upstream_fix breadcrumb NOW:\n.claude/scripts/logging/append_breadcrumb.sh \"{op_path}\" \"ttnn-kernel-writer-tdd\" '"'"'{\"event\":\"upstream_fix\",\"stage\":\"CURRENT_STAGE\",\"file\":\"FILENAME\",\"change\":\"WHAT_YOU_CHANGED\",\"reason\":\"WHY\"}'"'"'")
      }
    }'
    ;;
  *)
    # Not a relevant file, no reminder needed
    exit 0
    ;;
esac
