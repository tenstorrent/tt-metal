#!/bin/bash
# PostToolUseFailure hook for sfpu-operation-implementor: classify error on test failure
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

[[ "$COMMAND" == *pytest* || "$COMMAND" == *test* ]] || exit 0

ERROR=$(echo "$INPUT" | jq -r '.error // empty')

# Hang detection: exit code 2 or triage logs present
if [[ "$ERROR" == *"status code 2"* || "$ERROR" == *"exit code 2"* ]] || ls /tmp/tt-test-triage-*.log 2>/dev/null | xargs -I{} test -s {} 2>/dev/null; then
  jq -n '{
    hookSpecificOutput: {
      hookEventName: "PostToolUseFailure",
      additionalContext: "HANG DETECTED. You MUST: 1) Kill pytest: pkill -9 -f pytest || true. 2) Reset device: tt-smi -r. 3) Diagnose: common hang causes are missing SfpuType entry, wrong include guard macro name in get_macro_definition(), or wrong SFPU_OP_CHAIN_0 expansion. 4) Update the implementation notes ## Debug Log with this hang."
    }
  }'
else
  jq -n '{
    hookSpecificOutput: {
      hookEventName: "PostToolUseFailure",
      additionalContext: "Test FAILED. You MUST: 1) Classify the error: build_error (missing include, wrong macro), runtime_error (wrong bitcast, missing enum), numerical_error (low PCC, wrong SFPU logic), or assert_failure. 2) Diagnose the root cause before changing code. 3) Update the implementation notes ## Debug Log with this failure and your fix."
    }
  }'
fi
