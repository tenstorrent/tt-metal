#!/bin/bash
# PostToolUseFailure hook for sfpu-operation-tester: classify error and guide debugging
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

[[ "$COMMAND" == *pytest* || "$COMMAND" == *test* ]] || exit 0

ERROR=$(echo "$INPUT" | jq -r '.error // empty')

# Hang detection: exit code 2 or triage logs present
if [[ "$ERROR" == *"status code 2"* || "$ERROR" == *"exit code 2"* ]] || ls /tmp/tt-test-triage-*.log 2>/dev/null | xargs -I{} test -s {} 2>/dev/null; then
  jq -n '{
    hookSpecificOutput: {
      hookEventName: "PostToolUseFailure",
      additionalContext: "HANG DETECTED. You MUST: 1) Log a test_run breadcrumb (status=hang). 2) Kill hung process: pkill -9 -f pytest || true. 3) Diagnose: check SfpuType entry in llk_sfpu_types.h (BOTH architectures), check get_macro_definition() macro name matches sfpu_split_includes.h, check tile_init/tile function signatures match SFPU_OP_CHAIN_0 expectations. 4) Log a hypothesis breadcrumb BEFORE making any code changes. 5) Update implementation notes ## Debug Log."
    }
  }'
else
  jq -n '{
    hookSpecificOutput: {
      hookEventName: "PostToolUseFailure",
      additionalContext: "Test FAILED. You MUST: 1) Log a test_run breadcrumb (status=fail, include failure_type and error summary). 2) Classify: build_error (missing include, wrong macro), runtime_error (wrong bitcast, missing enum), numerical_error (low PCC, wrong SFPU logic), or assert_failure. 3) Log a hypothesis breadcrumb BEFORE making any code changes. 4) After fixing, log a fix_applied breadcrumb. 5) Update implementation notes ## Debug Log."
    }
  }'
fi
