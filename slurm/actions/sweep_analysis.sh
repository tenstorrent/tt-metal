#!/usr/bin/env bash
# sweep_analysis.sh - Run sweep analysis pipeline (parse, extract, push, notify).
# Equivalent to .github/actions/sweep-run-analysis/action.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") --artifacts-dir DIR [OPTIONS]

Run the sweep analysis pipeline: parse artifacts, extract results, optionally
push to a database, and send a Slack notification.

Required:
  --artifacts-dir DIR     Directory containing sweep run artifacts

Options:
  --output-dir DIR        Output directory for results (default: generated/sweep_results)
  --db-url URL            PostgreSQL connection string for sweep database
  --slack-webhook URL     Slack webhook URL for notifications
  --run-type TYPE         Sweep run type (nightly, comprehensive, etc.)
  --conclusion STATUS     Workflow conclusion (success, failure, cancelled)
  -h, --help              Show this help message

Environment:
  PIPELINE_ID             Pipeline identifier (auto-generated if unset)
  SWEEP_DB_URL            Fallback for --db-url
  SLACK_WEBHOOK_URL       Fallback for --slack-webhook
EOF
    exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

ARTIFACTS_DIR=""
OUTPUT_DIR="generated/sweep_results"
DB_URL="${SWEEP_DB_URL:-}"
SLACK_WEBHOOK="${SLACK_WEBHOOK_URL:-}"
RUN_TYPE=""
CONCLUSION=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --artifacts-dir) ARTIFACTS_DIR="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --db-url)        DB_URL="$2"; shift 2 ;;
        --slack-webhook) SLACK_WEBHOOK="$2"; shift 2 ;;
        --run-type)      RUN_TYPE="$2"; shift 2 ;;
        --conclusion)    CONCLUSION="$2"; shift 2 ;;
        -h|--help)       usage 0 ;;
        *)               log_error "Unknown option: $1"; usage 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ -z "${ARTIFACTS_DIR}" ]]; then
    log_error "--artifacts-dir is required"
    usage 1
fi

if [[ ! -d "${ARTIFACTS_DIR}" ]]; then
    log_fatal "Artifacts directory not found: ${ARTIFACTS_DIR}"
fi

# ---------------------------------------------------------------------------
# Locate Python scripts (local lib first, then GHA fallback)
# ---------------------------------------------------------------------------

SWEEP_LIB="${SCRIPT_DIR}/../lib/data_analysis"
SWEEP_SCRIPTS="${REPO_ROOT}/.github/actions/sweep-run-analysis/scripts"

find_script() {
    local name="$1"
    if [[ -f "${SWEEP_LIB}/${name}" ]]; then
        echo "${SWEEP_LIB}/${name}"
    elif [[ -f "${SWEEP_SCRIPTS}/${name}" ]]; then
        echo "${SWEEP_SCRIPTS}/${name}"
    else
        echo ""
    fi
}

PARSE_SCRIPT="$(find_script parse_sweep_artifacts.py)"
EXTRACT_SCRIPT="$(find_script extract_sweep_results.py)"
PUSH_SCRIPT="$(find_script push_sweep_results.py)"
NOTIFY_SCRIPT="$(find_script send_slack_notification.py)"

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

mkdir -p "${OUTPUT_DIR}"

log_info "=== Sweep analysis ==="
log_info "  Artifacts: ${ARTIFACTS_DIR}"
log_info "  Output:    ${OUTPUT_DIR}"

if [[ -n "${PARSE_SCRIPT}" ]]; then
    log_info "Step 1/4: Parsing sweep artifacts"
    python3 "${PARSE_SCRIPT}" \
        --input-dir "${ARTIFACTS_DIR}" \
        --output-dir "${OUTPUT_DIR}" || log_warn "Parse step failed (non-fatal)"
else
    log_warn "parse_sweep_artifacts.py not found; skipping"
fi

if [[ -n "${EXTRACT_SCRIPT}" ]]; then
    log_info "Step 2/4: Extracting sweep results"
    EXTRACT_ENV=()
    [[ -n "${DB_URL}" ]] && EXTRACT_ENV+=(DATABASE_URL="${DB_URL}")
    [[ -n "${PIPELINE_ID:-}" ]] && EXTRACT_ENV+=(SOURCE_GITHUB_RUN_ID="${PIPELINE_ID}")
    env "${EXTRACT_ENV[@]+"${EXTRACT_ENV[@]}"}" \
        python3 "${EXTRACT_SCRIPT}" \
            --input-dir "${OUTPUT_DIR}" \
            --output-dir "${OUTPUT_DIR}" || log_warn "Extract step failed (non-fatal)"
else
    log_warn "extract_sweep_results.py not found; skipping"
fi

if [[ -n "${DB_URL}" && -n "${PUSH_SCRIPT}" ]]; then
    log_info "Step 3/4: Pushing results to database"
    python3 "${PUSH_SCRIPT}" \
        --input-dir "${OUTPUT_DIR}" \
        --db-url "${DB_URL}" || log_warn "Push step failed (non-fatal)"
else
    log_info "Step 3/4: Skipped (no DB URL or push script)"
fi

if [[ -n "${SLACK_WEBHOOK}" && -n "${NOTIFY_SCRIPT}" ]]; then
    log_info "Step 4/4: Sending Slack notification"
    NOTIFY_ENV=(SLACK_WEBHOOK_URL="${SLACK_WEBHOOK}")
    [[ -n "${CONCLUSION}" ]] && NOTIFY_ENV+=(CONCLUSION="${CONCLUSION}")
    [[ -n "${PIPELINE_ID:-}" ]] && NOTIFY_ENV+=(SOURCE_GITHUB_RUN_ID="${PIPELINE_ID}")
    [[ -n "${RUN_TYPE}" ]] && NOTIFY_ENV+=(RUN_TYPE="${RUN_TYPE}")
    env "${NOTIFY_ENV[@]}" \
        python3 "${NOTIFY_SCRIPT}" \
            --input-dir "${OUTPUT_DIR}" \
            --webhook-url "${SLACK_WEBHOOK}" || log_warn "Notify step failed (non-fatal)"
else
    log_info "Step 4/4: Skipped (no webhook or notify script)"
fi

log_info "=== Sweep analysis complete ==="
