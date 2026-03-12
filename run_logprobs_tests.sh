#!/bin/bash
# Run all logprobs unit tests and save output to outcome.txt
#
# Usage:
#   ./run_logprobs_tests.sh
#
# Output: outcome.txt in the current directory

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_FILE="outcome.txt"
TEST_FILE="models/common/tests/test_sampling.py"

# All logprobs test functions
TESTS=(
    # Existing tests (sampled-token-only logprobs)
    "test_log_probs_with_sub_core_grids_on_galaxy"

    # New tests (top-K logprobs, TG Galaxy only)
    "test_top_k_log_probs_on_galaxy"
    "test_top_k_log_probs_returns_none_when_not_needed"
    "test_per_user_logprobs_enabled"
    "test_transfer_logprobs_to_host_response_format"
    "test_set_log_probs_mode_validation"
    "test_top_k_logprobs_pcc_torch_vs_tt"
)

echo "========================================" | tee "$OUTPUT_FILE"
echo "Logprobs Test Suite - $(date)" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

for test_name in "${TESTS[@]}"; do
    TOTAL=$((TOTAL + 1))
    echo "--- Running: ${test_name} ---" | tee -a "$OUTPUT_FILE"

    pytest "${TEST_FILE}::${test_name}" -v -s --timeout=120 2>&1 | tee -a "$OUTPUT_FILE"
    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        PASSED=$((PASSED + 1))
        echo ">>> RESULT: PASSED" | tee -a "$OUTPUT_FILE"
    elif [ $EXIT_CODE -eq 5 ]; then
        # pytest exit code 5 = no tests collected (skipped)
        SKIPPED=$((SKIPPED + 1))
        echo ">>> RESULT: SKIPPED (not Galaxy)" | tee -a "$OUTPUT_FILE"
    else
        FAILED=$((FAILED + 1))
        echo ">>> RESULT: FAILED (exit code $EXIT_CODE)" | tee -a "$OUTPUT_FILE"
    fi
    echo "" | tee -a "$OUTPUT_FILE"
done

echo "========================================" | tee -a "$OUTPUT_FILE"
echo "SUMMARY: ${TOTAL} tests | ${PASSED} passed | ${FAILED} failed | ${SKIPPED} skipped" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
