#!/bin/bash
# Streaming vsa_sdpa suite under the safe-pytest harness (dispatch-timeout hang detection).
cd "$(dirname "$0")"
LOG=/tmp/claude-4015/-data-cglagovich-tt-metal/7bf353e5-b3ae-491a-987f-f299bc56b26a/scratchpad/suite_safe.log
./scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/sdpa/test_vsa_sdpa.py -q -k stream > "$LOG" 2>&1
code=$?
echo "SAFE_EXIT $code"
grep -E "passed|failed|Hang|HANG|SAFE_PYTEST" "$LOG" | tail -8
