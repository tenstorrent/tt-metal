#!/bin/bash
# PostToolUse hook for kernel-writer: remind agent to log + commit after test pass
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

[[ "$COMMAND" == *test* ]] || exit 0

jq -n '{
  hookSpecificOutput: {
    hookEventName: "PostToolUse",
    additionalContext: "Tests PASSED. You MUST: 1) Log a test_run breadcrumb (status=pass, include test path and params). 2) Commit changes now with a descriptive message before continuing."
  }
}'
