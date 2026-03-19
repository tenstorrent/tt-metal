#!/bin/bash
# PostToolUse hook for sfpu-operation-implementor: remind agent after test pass
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

[[ "$COMMAND" == *pytest* || "$COMMAND" == *test* ]] || exit 0

jq -n '{
  hookSpecificOutput: {
    hookEventName: "PostToolUse",
    additionalContext: "Tests PASSED. You MUST: 1) Report the PCC values achieved. 2) Update the implementation notes with a ## Debug Log entry for this test run. 3) Do NOT commit — the orchestrator handles commits."
  }
}'
