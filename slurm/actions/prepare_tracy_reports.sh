#!/usr/bin/env bash
# prepare_tracy_reports.sh - Clean up intermediate Tracy files and rename CSV reports.
# Equivalent to .github/actions/prepare-tracy-reports/action.yml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") --report-dir DIR [OPTIONS]

Clean up large intermediate Tracy profiler files and rename CSV reports
with the model name for easier identification.

Required:
  --report-dir DIR      Profiler directory containing Tracy output

Options:
  --output-dir DIR      Destination for prepared reports (default: <report-dir>/reports)
  --model-name NAME     Model name to include in renamed CSV files
  --card CARD           Card type suffix (e.g. N150, N300)
  -h, --help            Show this help message

Environment:
  PIPELINE_ID           Pipeline identifier (auto-generated if unset)
EOF
    exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

REPORT_DIR=""
OUTPUT_DIR=""
MODEL_NAME=""
CARD=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --report-dir)  REPORT_DIR="$2"; shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --model-name)  MODEL_NAME="$2"; shift 2 ;;
        --card)        CARD="$2"; shift 2 ;;
        -h|--help)     usage 0 ;;
        *)             log_error "Unknown option: $1"; usage 1 ;;
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

[[ -z "${OUTPUT_DIR}" ]] && OUTPUT_DIR="${REPORT_DIR}/reports"

# ---------------------------------------------------------------------------
# Clean up large intermediate files
# ---------------------------------------------------------------------------

log_info "Preparing Tracy reports: ${REPORT_DIR} -> ${OUTPUT_DIR}"

find "${REPORT_DIR}" -name "profile_log_device.csv" -type f -delete 2>/dev/null || true
find "${REPORT_DIR}" -name "tracy_profile_log_host.tracy" -type f -delete 2>/dev/null || true

# ---------------------------------------------------------------------------
# Rename CSV files with model name
# ---------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}"
REPORT_COUNT=0

REPORTS_SUBDIR="${REPORT_DIR}/reports"
if [[ -d "${REPORTS_SUBDIR}" ]]; then
    while IFS= read -r -d '' csv; do
        dir="$(dirname "${csv}")"
        if [[ -n "${MODEL_NAME}" ]]; then
            if [[ -n "${CARD}" ]]; then
                dest="${dir}/ops_perf_results_${MODEL_NAME}_${CARD}.csv"
            else
                dest="${dir}/ops_perf_results_${MODEL_NAME}.csv"
            fi
            mv "${csv}" "${dest}"
        fi
        REPORT_COUNT=$((REPORT_COUNT + 1))
    done < <(find "${REPORTS_SUBDIR}" -name 'ops_perf_results_*.csv' -type f -print0)
fi

# ---------------------------------------------------------------------------
# Copy Tracy files to output
# ---------------------------------------------------------------------------

for trace_file in "${REPORT_DIR}"/*.tracy "${REPORT_DIR}"/**/*.tracy; do
    [[ -f "${trace_file}" ]] || continue
    base="$(basename "${trace_file}" .tracy)"
    dest="${OUTPUT_DIR}/${base}"
    mkdir -p "${dest}"
    cp "${trace_file}" "${dest}/"
    REPORT_COUNT=$((REPORT_COUNT + 1))

    if command -v tracy-csvexport &>/dev/null; then
        tracy-csvexport "${trace_file}" > "${dest}/${base}.csv" 2>/dev/null || true
    fi
done

log_info "Prepared ${REPORT_COUNT} Tracy report(s)"

if is_slurm_job && (( REPORT_COUNT > 0 )); then
    stage_test_report "${PIPELINE_ID}" "tracy-$(get_job_name)" "${OUTPUT_DIR}"
fi
