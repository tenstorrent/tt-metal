#!/usr/bin/env bash
# single_card_perf_test.sh - Run single-card performance tests.
# Equivalent to .github/actions/single-card-perf-test/action.yml
#
# Usage: single_card_perf_test.sh [--test-suite SUITE] [--arch ARCH] [--timeout SECS]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_config env

TEST_SUITE="${TEST_SUITE:-perf}"
ARCH="${ARCH_NAME:-wormhole_b0}"
TIMEOUT="${TIMEOUT:-3600}"
RESULTS_DIR="${RESULTS_DIR:-generated/perf_reports}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test-suite) TEST_SUITE="$2"; shift 2 ;;
        --arch)       ARCH="$2"; shift 2 ;;
        --timeout)    TIMEOUT="$2"; shift 2 ;;
        *)            log_warn "Unknown option: $1"; shift ;;
    esac
done

log_info "=== Single-card perf test ==="
log_info "  Suite:   ${TEST_SUITE}"
log_info "  Arch:    ${ARCH}"
log_info "  Timeout: ${TIMEOUT}s"

mkdir -p "${RESULTS_DIR}"

LOG_FILE="${RESULTS_DIR}/perf_test_${TEST_SUITE}.log"

log_info "Running perf tests"
timeout "${TIMEOUT}" pytest tests/perf/"${TEST_SUITE}" \
    --arch "${ARCH}" \
    --json-report --json-report-file="${RESULTS_DIR}/results.json" \
    -v 2>&1 | tee "${LOG_FILE}" || {
    rc=$?
    log_error "Perf tests exited with code ${rc}"
}

if [[ -f "${RESULTS_DIR}/results.json" ]]; then
    log_info "Generating summary from results"
    python3 -c "
import json, sys
data = json.load(open('${RESULTS_DIR}/results.json'))
total = data.get('summary', {}).get('total', 0)
passed = data.get('summary', {}).get('passed', 0)
failed = data.get('summary', {}).get('failed', 0)
print(f'Total: {total}, Passed: {passed}, Failed: {failed}')
" 2>/dev/null || true
fi

if is_slurm_job; then
    stage_test_report "${PIPELINE_ID}" "$(get_job_name)" "${RESULTS_DIR}"
fi

log_info "=== Single-card perf test complete ==="
