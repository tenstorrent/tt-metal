#!/usr/bin/env bash
# aggregate_ops_report.sh - Aggregate Tracy profiler CSV reports across jobs.
# Equivalent to .github/actions/aggregate-ops-report/action.yml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") --report-dirs DIR1,DIR2... --output FILE [OPTIONS]

Aggregate Tracy profiler CSV reports (ops_perf_results_*.csv) from one or more
directories and produce a combined summary.

Required:
  --report-dirs DIRS    Comma-separated list of directories containing CSV reports
  --output FILE         Path for the aggregated output file

Options:
  -h, --help            Show this help message

Environment:
  PIPELINE_ID           Pipeline identifier (auto-generated if unset)
EOF
    exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

REPORT_DIRS=""
OUTPUT_FILE=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --report-dirs) REPORT_DIRS="$2"; shift 2 ;;
        --output)      OUTPUT_FILE="$2"; shift 2 ;;
        -h|--help)     usage 0 ;;
        *)             log_error "Unknown option: $1"; usage 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ -z "${REPORT_DIRS}" ]]; then
    log_error "--report-dirs is required"
    usage 1
fi

if [[ -z "${OUTPUT_FILE}" ]]; then
    log_error "--output is required"
    usage 1
fi

# ---------------------------------------------------------------------------
# Locate the aggregation script
# ---------------------------------------------------------------------------

AGGREGATE_SCRIPT="${SCRIPT_DIR}/../lib/aggregate_ops_report.py"
[[ -f "${AGGREGATE_SCRIPT}" ]] || \
    AGGREGATE_SCRIPT="${REPO_ROOT}/.github/actions/aggregate-ops-report/aggregate_ops.py"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

IFS=',' read -ra DIRS <<< "${REPORT_DIRS}"

log_info "Aggregating ops reports from ${#DIRS[@]} director(ies)"
for d in "${DIRS[@]}"; do
    log_info "  ${d}"
done

mkdir -p "$(dirname "${OUTPUT_FILE}")"

if [[ -f "${AGGREGATE_SCRIPT}" ]]; then
    export REPORTS_PATH="${REPORT_DIRS}"
    export OUTPUT_PATH="$(dirname "${OUTPUT_FILE}")"
    python3 "${AGGREGATE_SCRIPT}"
else
    log_info "Aggregate script not found; performing simple CSV merge"

    HEADER_WRITTEN=false
    for dir in "${DIRS[@]}"; do
        [[ -d "${dir}" ]] || { log_warn "Directory not found: ${dir}"; continue; }
        while IFS= read -r -d '' csv; do
            if [[ "${HEADER_WRITTEN}" == false ]]; then
                head -1 "${csv}" > "${OUTPUT_FILE}"
                HEADER_WRITTEN=true
            fi
            tail -n +2 "${csv}" >> "${OUTPUT_FILE}"
        done < <(find "${dir}" -name 'ops_perf_results_*.csv' -type f -print0)
    done
fi

if [[ -f "${OUTPUT_FILE}" ]]; then
    LINE_COUNT="$(wc -l < "${OUTPUT_FILE}")"
    log_info "Aggregated report: ${OUTPUT_FILE} (${LINE_COUNT} lines)"
else
    log_warn "No ops reports found to aggregate"
fi
