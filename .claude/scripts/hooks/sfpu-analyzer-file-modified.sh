#!/bin/bash
# PostToolUse hook for sfpu-operation-analyzer: remind agent to log breadcrumb after writing analysis
INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool_name // empty')

[[ "$TOOL" == "Write" ]] || exit 0

FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

case "$FILE_PATH" in
  *_analysis.md|*_sfpu_analysis.md|*_analysis-*.md|*_sfpu_analysis-*.md)
    jq -n --arg file "$FILE_PATH" '{
      hookSpecificOutput: {
        hookEventName: "PostToolUse",
        additionalContext: ("You just wrote an analysis file: " + $file + ". You MUST: 1) Log an analysis_written breadcrumb with the file path and sections completed. 2) Proceed to verification (grep SFPU identifiers) and then log the complete breadcrumb.")
      }
    }'
    ;;
  *)
    exit 0
    ;;
esac
