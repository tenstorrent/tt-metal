#!/bin/bash
# PostToolUse hook for sfpu-operation-tester: remind agent after test pass
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

[[ "$COMMAND" == *pytest* || "$COMMAND" == *test* ]] || exit 0

jq -n '{
  hookSpecificOutput: {
    hookEventName: "PostToolUse",
    additionalContext: "Tests PASSED. You MUST: 1) Log a test_run breadcrumb (status=pass, include max ULP and allclose results). 2) Update the implementation notes with a ## Test Results section. 3) Log a complete breadcrumb. 4) Do NOT commit — the orchestrator handles commits."
  }
}'
