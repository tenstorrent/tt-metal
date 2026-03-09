#!/bin/bash
# eval_test_runner.sh - Run golden tests with structured output for eval tracking.
#
# Wraps pytest with:
# - JUnit XML output for per-test results
# - TT_METAL_OPERATION_TIMEOUT_SECONDS for hang detection
# - Hang plugin to skip remaining parametrizations after hang
# - Failure classification into categories
# - Device reset after hangs
# - flock for device serialization
#
# Usage: eval/eval_test_runner.sh <test_dir> <output_dir>
#
# Outputs:
#   <output_dir>/junit.xml          - Raw pytest JUnit XML
#   <output_dir>/test_results.json  - Classified per-test results
#   <output_dir>/golden_results.txt - Summary line

set -o pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="/tmp/tt-device.lock"
DISPATCH_TIMEOUT=5

if [[ $# -lt 2 ]]; then
    echo "Usage: eval/eval_test_runner.sh <test_dir> <output_dir>" >&2
    exit 1
fi

TEST_DIR="$1"
OUTPUT_DIR="$2"
mkdir -p "$OUTPUT_DIR"

JUNIT_XML="${OUTPUT_DIR}/junit.xml"
TEST_RESULTS_JSON="${OUTPUT_DIR}/test_results.json"
GOLDEN_RESULTS="${OUTPUT_DIR}/golden_results.txt"

# --- Setup environment ---
cd "$REPO_DIR"
if [[ -f python_env/bin/activate ]]; then
    source python_env/bin/activate
fi

export TT_METAL_OPERATION_TIMEOUT_SECONDS="$DISPATCH_TIMEOUT"

# --- Pre-flight: validate API contract (no device needed) ---
CONTRACT_FILE="${TEST_DIR}/api_contract.md"
if [[ -f "$CONTRACT_FILE" ]]; then
    echo "EVAL_RUNNER: Validating API contract..." >&2
    CONTRACT_RESULT="${OUTPUT_DIR}/contract_validation.json"
    python3 -m eval.validate_contract "${TEST_DIR}" > "$CONTRACT_RESULT" 2>&1
    CONTRACT_EXIT=$?
    if [[ $CONTRACT_EXIT -ne 0 ]]; then
        echo "EVAL_RUNNER: API CONTRACT VALIDATION FAILED" >&2
        cat "$CONTRACT_RESULT" >&2
        echo "EVAL_RUNNER: Fix the operation signature before running device tests" >&2
        # Still continue to run tests — they'll fail with ImportError/TypeError
        # but the contract report gives a clearer diagnosis
    else
        echo "EVAL_RUNNER: API contract validated OK" >&2
    fi
fi

# --- Acquire device lock ---
exec 9>"$LOCK_FILE"
echo "EVAL_RUNNER: Waiting for device lock..." >&2
flock 9
echo "EVAL_RUNNER: Device lock acquired" >&2

# --- Run pytest with JUnit XML and hang plugin ---
echo "EVAL_RUNNER: Running tests in ${TEST_DIR}..." >&2
pytest "${TEST_DIR}" \
    --junitxml="${JUNIT_XML}" \
    -p eval.hang_plugin \
    --tb=short \
    -q \
    > "${OUTPUT_DIR}/pytest_stdout.log" 2>&1 || true

# Release device lock early (classification doesn't need device)
exec 9>&-

if [[ ! -f "$JUNIT_XML" ]]; then
    echo "EVAL_RUNNER: ERROR - No JUnit XML produced" >&2
    echo "PASSED=0 FAILED=0 ERRORS=0 SKIPPED=0 HANGS=0 TOTAL=0" > "$GOLDEN_RESULTS"
    echo "[]" > "$TEST_RESULTS_JSON"
    exit 1
fi

# --- Classify failures and produce summary ---
python3 -c "
import json, sys
sys.path.insert(0, '${REPO_DIR}')
from eval.classify_failures import parse_junit_xml
from pathlib import Path

results = parse_junit_xml(Path('${JUNIT_XML}'))
Path('${TEST_RESULTS_JSON}').write_text(json.dumps(results, indent=2))

passed = sum(1 for r in results if r['status'] == 'passed')
failed = sum(1 for r in results if r['status'] == 'failed')
errors = sum(1 for r in results if r['status'] == 'error')
skipped = sum(1 for r in results if r['status'] == 'skipped')
hangs = sum(1 for r in results if r.get('failure_category') == 'hang')
total = len(results)

with open('${GOLDEN_RESULTS}', 'w') as f:
    f.write(f'PASSED={passed} FAILED={failed} ERRORS={errors} SKIPPED={skipped} HANGS={hangs} TOTAL={total}\n')

print(f'EVAL_RUNNER: {passed}/{total} passed ({failed} failed, {errors} errors, {skipped} skipped, {hangs} hangs)', file=sys.stderr)
"

# --- Reset device if any hang occurred ---
HANGS="$(grep -oP 'HANGS=\K\d+' "${GOLDEN_RESULTS}" || echo 0)"
if [[ "$HANGS" -gt 0 ]]; then
    echo "EVAL_RUNNER: Hang detected, resetting device..." >&2
    tt-smi -r || echo "EVAL_RUNNER: WARNING - device reset failed" >&2
    sleep 2
    echo "EVAL_RUNNER: Device reset complete" >&2
fi
