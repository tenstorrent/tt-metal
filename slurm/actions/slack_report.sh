#!/usr/bin/env bash
# slack_report.sh - Unified Slack reporter for pipeline notifications.
# Equivalent to .github/actions/slack-report/action.yaml and the
# slack-report-merge-gate, slack-report-merge-gate-main,
# slack-report-analyze-workflow-data variants.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib notify

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") --type TYPE [OPTIONS]

Send a Slack notification for a pipeline event.  Combines the four GitHub
Actions slack-report variants into a single script driven by --type.

Required:
  --type TYPE           Report type: failure, success, merge-gate, or summary

Options:
  --webhook-url URL     Slack webhook URL (or set SLACK_WEBHOOK_URL)
  --channel-id ID       Slack channel ID or name
  --pipeline-id ID      Pipeline identifier (default: \$PIPELINE_ID)
  --job-name NAME       Job name for context (default: Slurm job name)
  --message MSG         Custom message body (overrides generated text)
  --owner ID            Slack user ID to mention (for failure/merge-gate)
  --results-file FILE   Job results file for summary type (job_name:exit_code per line)
  -h, --help            Show this help message

Environment:
  SLACK_WEBHOOK_URL     Fallback for --webhook-url
  PIPELINE_ID           Fallback for --pipeline-id
EOF
    exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

REPORT_TYPE=""
WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
CHANNEL_ID=""
QUERY_PIPELINE="${PIPELINE_ID:-}"
JOB_NAME="$(get_job_name)"
MESSAGE=""
OWNER=""
RESULTS_FILE=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)         REPORT_TYPE="$2"; shift 2 ;;
        --webhook-url)  WEBHOOK_URL="$2"; shift 2 ;;
        --channel-id)   CHANNEL_ID="$2"; shift 2 ;;
        --pipeline-id)  QUERY_PIPELINE="$2"; shift 2 ;;
        --job-name)     JOB_NAME="$2"; shift 2 ;;
        --message)      MESSAGE="$2"; shift 2 ;;
        --owner)        OWNER="$2"; shift 2 ;;
        --results-file) RESULTS_FILE="$2"; shift 2 ;;
        -h|--help)      usage 0 ;;
        *)              log_error "Unknown option: $1"; usage 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ -z "${REPORT_TYPE}" ]]; then
    log_error "--type is required"
    usage 1
fi

case "${REPORT_TYPE}" in
    failure|success|merge-gate|summary) ;;
    *) log_error "Invalid type '${REPORT_TYPE}'; expected failure|success|merge-gate|summary"; usage 1 ;;
esac

if [[ -z "${WEBHOOK_URL}" ]]; then
    log_warn "No webhook URL provided; skipping Slack notification"
    exit 0
fi

# ---------------------------------------------------------------------------
# Build payload per type
# ---------------------------------------------------------------------------

PAYLOAD=""

case "${REPORT_TYPE}" in
    failure)
        if [[ -n "${MESSAGE}" ]]; then
            PAYLOAD="${MESSAGE}"
        else
            OWNER_TAG=""
            [[ -n "${OWNER}" ]] && OWNER_TAG="<@${OWNER}> "
            PAYLOAD="${OWNER_TAG}:x: *${JOB_NAME}* failed"
            PAYLOAD+="\nPipeline: ${QUERY_PIPELINE}"
            PAYLOAD+="\nRef: ${GIT_REF} (\`${GIT_SHORT_SHA}\`)"
        fi
        send_slack_message "${WEBHOOK_URL}" "${PAYLOAD}"
        ;;

    success)
        if [[ -n "${MESSAGE}" ]]; then
            PAYLOAD="${MESSAGE}"
        else
            PAYLOAD=":white_check_mark: *${JOB_NAME}* succeeded"
            PAYLOAD+="\nPipeline: ${QUERY_PIPELINE}"
            PAYLOAD+="\nRef: ${GIT_REF} (\`${GIT_SHORT_SHA}\`)"
        fi
        send_slack_message "${WEBHOOK_URL}" "${PAYLOAD}"
        ;;

    merge-gate)
        if [[ -n "${MESSAGE}" ]]; then
            PAYLOAD="${MESSAGE}"
        else
            OWNER_TAG=""
            [[ -n "${OWNER}" ]] && OWNER_TAG="<@${OWNER}> "
            PAYLOAD="${OWNER_TAG}:rotating_light: Merge-gate failure: *${JOB_NAME}*"
            PAYLOAD+="\nPipeline: ${QUERY_PIPELINE}"
            PAYLOAD+="\nRef: ${GIT_REF} (\`${GIT_SHORT_SHA}\`)"
        fi
        send_slack_message "${WEBHOOK_URL}" "${PAYLOAD}"
        ;;

    summary)
        if [[ -n "${RESULTS_FILE}" && -f "${RESULTS_FILE}" ]]; then
            send_pipeline_summary "${WEBHOOK_URL}" "${QUERY_PIPELINE}" "${RESULTS_FILE}"
        elif [[ -n "${MESSAGE}" ]]; then
            send_slack_message "${WEBHOOK_URL}" "${MESSAGE}"
        else
            log_error "Summary type requires --results-file or --message"
            exit 1
        fi
        ;;
esac

log_info "Slack notification sent: type=${REPORT_TYPE} job=${JOB_NAME}"
