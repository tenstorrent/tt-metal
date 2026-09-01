#!/bin/bash
# Suite (correctness) then bench, both under safe_pytest; logs to scratchpad.
cd "$(dirname "$0")"
SP=/tmp/claude-4015/-data-cglagovich-tt-metal/7bf353e5-b3ae-491a-987f-f299bc56b26a/scratchpad
./run_vsa_suite_safe.sh
if ! grep -q "SAFE_PYTEST_RESULT: PASS" "$SP/suite_safe.log"; then
    echo "SUITE_FAILED" > "$SP/bench_v6c.log"
    exit 1
fi
./scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/sdpa/test_vsa_sdpa_perf.py -q -s > "$SP/bench_v6c.log" 2>&1
echo DONE_ALL >> "$SP/bench_v6c.log"
