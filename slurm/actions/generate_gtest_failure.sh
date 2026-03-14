#!/usr/bin/env bash
# generate_gtest_failure.sh - Parse gtest XML reports and print failure annotations.
# Equivalent to .github/actions/generate-gtest-failure-message/action.yml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") --report-dir DIR [OPTIONS]

Parse gtest XML test reports and print failure annotations.  Exits non-zero
when failures are found.

Required:
  --report-dir DIR    Directory containing gtest XML report files

Options:
  --format FORMAT     Output format: text (default) or json
  -h, --help          Show this help message

Environment:
  GTEST_REPORT_DIR    Fallback for --report-dir
EOF
    exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

REPORT_DIR="${GTEST_REPORT_DIR:-}"
FORMAT="text"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --report-dir) REPORT_DIR="$2"; shift 2 ;;
        --format)     FORMAT="$2"; shift 2 ;;
        -h|--help)    usage 0 ;;
        *)            log_error "Unknown option: $1"; usage 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ -z "${REPORT_DIR}" ]]; then
    log_error "--report-dir is required"
    usage 1
fi

if [[ ! -d "${REPORT_DIR}" ]]; then
    log_warn "Report directory does not exist: ${REPORT_DIR}"
    exit 0
fi

case "${FORMAT}" in
    text|json) ;;
    *) log_error "Invalid format '${FORMAT}'; expected text or json"; usage 1 ;;
esac

# ---------------------------------------------------------------------------
# Locate the annotation script
# ---------------------------------------------------------------------------

FAILURE_SCRIPT="${SCRIPT_DIR}/../lib/print_gtest_annotations.py"
[[ -f "${FAILURE_SCRIPT}" ]] || \
    FAILURE_SCRIPT="${REPO_ROOT}/.github/actions/generate-gtest-failure-message/print_gtest_annotations.py"

if [[ -f "${FAILURE_SCRIPT}" ]]; then
    log_info "Parsing gtest failures in ${REPORT_DIR} (format: ${FORMAT})"
    set +e
    python3 "${FAILURE_SCRIPT}" "${REPORT_DIR}"
    rc=$?
    set -e
    exit "${rc}"
fi

# ---------------------------------------------------------------------------
# Fallback: parse XML directly
# ---------------------------------------------------------------------------

log_info "Scanning for gtest failures in ${REPORT_DIR} (format: ${FORMAT})"

TOTAL_FAILURES=0
declare -a FAILURE_RECORDS=()

for xml_file in "${REPORT_DIR}"/*.xml; do
    [[ -f "${xml_file}" ]] || continue

    failures="$(grep -c '<failure' "${xml_file}" 2>/dev/null || echo 0)"
    if (( failures > 0 )); then
        TOTAL_FAILURES=$((TOTAL_FAILURES + failures))
        if [[ "${FORMAT}" == "json" ]]; then
            FAILURE_RECORDS+=("{\"file\": \"${xml_file}\", \"count\": ${failures}}")
        else
            echo "--- ${xml_file} (${failures} failures) ---"
            grep -A 5 '<failure' "${xml_file}" | head -50
            echo ""
        fi
    fi
done

if [[ "${FORMAT}" == "json" ]]; then
    printf '{"total_failures": %d, "files": [%s]}\n' \
        "${TOTAL_FAILURES}" \
        "$(IFS=,; echo "${FAILURE_RECORDS[*]+"${FAILURE_RECORDS[*]}"}")"
fi

if (( TOTAL_FAILURES > 0 )); then
    log_error "Total gtest failures: ${TOTAL_FAILURES}"
    exit 1
fi

log_info "No gtest failures found"
