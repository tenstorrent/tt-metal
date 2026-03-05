#!/usr/bin/env bash
# check_latency_bw.sh - Validate latency/bandwidth test results against thresholds.
# Equivalent to .github/actions/check-latency-bw-results/action.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") --results-file FILE [OPTIONS]

Validate fabric latency and bandwidth measurements against threshold values.
Each row in the results CSV is checked; latency must be below threshold and
bandwidth must be above threshold (both adjusted by --tolerance).

Required:
  --results-file FILE       CSV file with columns: test_name,metric_type,measured,threshold

Options:
  --thresholds-file FILE    External thresholds CSV to merge (overrides per-row thresholds)
  --tolerance PERCENT       Acceptable deviation percentage (default: 5)
  -h, --help                Show this help message

Environment:
  RESULTS_FILE              Fallback for --results-file
  TOLERANCE                 Fallback for --tolerance
EOF
    exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

RESULTS_FILE="${RESULTS_FILE:-}"
THRESHOLDS_FILE=""
TOLERANCE="${TOLERANCE:-5}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-file)    RESULTS_FILE="$2"; shift 2 ;;
        --thresholds-file) THRESHOLDS_FILE="$2"; shift 2 ;;
        --tolerance)       TOLERANCE="$2"; shift 2 ;;
        -h|--help)         usage 0 ;;
        *)                 log_error "Unknown option: $1"; usage 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ -z "${RESULTS_FILE}" ]]; then
    log_error "--results-file is required"
    usage 1
fi

if [[ ! -f "${RESULTS_FILE}" ]]; then
    log_fatal "Results file not found: ${RESULTS_FILE}"
fi

if [[ -n "${THRESHOLDS_FILE}" && ! -f "${THRESHOLDS_FILE}" ]]; then
    log_fatal "Thresholds file not found: ${THRESHOLDS_FILE}"
fi

# ---------------------------------------------------------------------------
# Load external thresholds (if provided)
# ---------------------------------------------------------------------------

declare -A EXT_THRESHOLDS=()

if [[ -n "${THRESHOLDS_FILE}" ]]; then
    while IFS=',' read -r t_name t_type t_value; do
        [[ "${t_name}" == "test_name" ]] && continue
        [[ -z "${t_name}" ]] && continue
        EXT_THRESHOLDS["${t_name}:${t_type}"]="${t_value}"
    done < "${THRESHOLDS_FILE}"
    log_info "Loaded ${#EXT_THRESHOLDS[@]} external threshold(s)"
fi

# ---------------------------------------------------------------------------
# Validate results
# ---------------------------------------------------------------------------

log_info "Validating latency/bandwidth results"
log_info "  Results:    ${RESULTS_FILE}"
log_info "  Thresholds: ${THRESHOLDS_FILE:-per-row}"
log_info "  Tolerance:  ${TOLERANCE}%"

FAILED=0

while IFS=',' read -r test_name metric_type measured threshold; do
    [[ "${test_name}" == "test_name" ]] && continue
    [[ -z "${test_name}" ]] && continue

    # Override with external threshold when available
    ext_key="${test_name}:${metric_type}"
    if [[ -n "${EXT_THRESHOLDS["${ext_key}"]:-}" ]]; then
        threshold="${EXT_THRESHOLDS["${ext_key}"]}"
    fi

    if [[ -z "${threshold}" || "${threshold}" == "null" ]]; then
        continue
    fi

    case "${metric_type}" in
        latency)
            limit="$(awk "BEGIN{printf \"%.2f\", ${threshold} * (1 + ${TOLERANCE}/100)}")"
            if awk "BEGIN{exit (${measured} <= ${limit}) ? 0 : 1}"; then
                log_info "PASS: ${test_name} latency=${measured} <= ${limit}"
            else
                log_error "FAIL: ${test_name} latency=${measured} > ${limit} (threshold=${threshold})"
                FAILED=$((FAILED + 1))
            fi
            ;;
        bandwidth)
            min_bw="$(awk "BEGIN{printf \"%.2f\", ${threshold} * (1 - ${TOLERANCE}/100)}")"
            if awk "BEGIN{exit (${measured} >= ${min_bw}) ? 0 : 1}"; then
                log_info "PASS: ${test_name} bw=${measured} >= ${min_bw}"
            else
                log_error "FAIL: ${test_name} bw=${measured} < ${min_bw} (threshold=${threshold})"
                FAILED=$((FAILED + 1))
            fi
            ;;
    esac
done < "${RESULTS_FILE}"

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

if (( FAILED > 0 )); then
    log_error "${FAILED} latency/bandwidth check(s) failed"
    exit 1
fi

log_info "All latency/bandwidth checks passed"
