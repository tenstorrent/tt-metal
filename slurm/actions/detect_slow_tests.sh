#!/usr/bin/env bash
# detect_slow_tests.sh - Scan test reports for tests exceeding a duration threshold.
# Equivalent to .github/actions/detect-slow-tests/action.yml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") --test-report-dir DIR [OPTIONS]

Scan test report XML/JSON files and fail if any individual test exceeds the
given duration threshold.

Required:
  --test-report-dir DIR   Directory containing test report files

Options:
  --threshold SECONDS     Maximum allowed test duration (default: 5.0)
  -h, --help              Show this help message

Environment:
  TEST_REPORT_DIR         Fallback for --test-report-dir
  SLOW_TEST_THRESHOLD     Fallback for --threshold
EOF
    exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

REPORT_DIR="${TEST_REPORT_DIR:-}"
THRESHOLD="${SLOW_TEST_THRESHOLD:-5.0}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test-report-dir) REPORT_DIR="$2"; shift 2 ;;
        --threshold)       THRESHOLD="$2"; shift 2 ;;
        -h|--help)         usage 0 ;;
        *)                 log_error "Unknown option: $1"; usage 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ -z "${REPORT_DIR}" ]]; then
    log_error "--test-report-dir is required"
    usage 1
fi

if [[ ! -d "${REPORT_DIR}" ]]; then
    log_warn "Test report directory does not exist: ${REPORT_DIR}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Locate the detection script
# ---------------------------------------------------------------------------

DETECT_SCRIPT="${SCRIPT_DIR}/../lib/detect_slow_tests.py"
[[ -f "${DETECT_SCRIPT}" ]] || \
    DETECT_SCRIPT="${REPO_ROOT}/.github/actions/detect-slow-tests/detect-slow-tests.py"

if [[ ! -f "${DETECT_SCRIPT}" ]]; then
    log_warn "detect-slow-tests.py not found; skipping"
    exit 0
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

log_info "Scanning for slow tests (threshold: ${THRESHOLD}s)"
log_info "  Report dir: ${REPORT_DIR}"

python3 "${DETECT_SCRIPT}" "${REPORT_DIR}" "${THRESHOLD}"
rc=$?

if (( rc != 0 )); then
    log_error "Slow tests detected (exit ${rc})"
    exit "${rc}"
fi

log_info "No slow tests found"
