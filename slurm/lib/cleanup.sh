#!/usr/bin/env bash
# cleanup.sh - Job epilogue: report staging, workspace teardown, notifications
# Equivalent to .github/actions/cleanup/action.yml

set -euo pipefail

# Guard against double-sourcing
[[ -n "${_SLURM_CI_CLEANUP_SH:-}" ]] && return 0
_SLURM_CI_CLEANUP_SH=1

SLURM_CI_LIB_DIR="${SLURM_CI_LIB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SLURM_CI_LIB_DIR}/common.sh"
source "${SLURM_CI_LIB_DIR}/artifacts.sh"

# ---------------------------------------------------------------------------
# cleanup_job - Main epilogue function
# ---------------------------------------------------------------------------
# Controlled by environment variables:
#   PIPELINE_ID             (required) Pipeline identifier
#   JOB_WORKSPACE           Working directory to remove (default: /work)
#   TEST_REPORT_DIR         Directory containing test reports (optional)
#   NOTIFY_ON_FAILURE       Set to 1 to send Slack notification on non-zero exit
#   SLACK_WEBHOOK_URL       Slack webhook URL (required if NOTIFY_ON_FAILURE=1)
cleanup_job() {
    local exit_code="${1:-$?}"
    local workspace="${JOB_WORKSPACE:-${CONTAINER_WORKDIR:-/work}}"
    local job_name="${SLURM_CI_JOB_NAME}"

    log_info "=== Job cleanup starting (exit_code=${exit_code}) ==="

    # -- Stage test reports if present --
    local report_dir="${TEST_REPORT_DIR:-${workspace}/generated/test_reports}"
    if [[ -d "$report_dir" ]]; then
        log_info "Staging test reports from ${report_dir}"
        stage_test_report "${PIPELINE_ID}" "$job_name" "$report_dir" || \
            log_warn "Failed to stage test reports"
    else
        log_debug "No test report directory found at ${report_dir}"
    fi

    # -- Remove workspace --
    if [[ -d "$workspace" ]]; then
        log_info "Removing workspace: ${workspace}"
        rm -rf "$workspace"
        log_info "Workspace removed"
    fi

    # -- Slack notification on failure --
    if [[ "${NOTIFY_ON_FAILURE:-0}" == "1" && "$exit_code" != "0" ]]; then
        _send_failure_notification "$exit_code" "$job_name"
    fi

    log_info "=== Job cleanup complete ==="
}

# ---------------------------------------------------------------------------
# _send_failure_notification - POST to Slack webhook on failure
# ---------------------------------------------------------------------------
_send_failure_notification() {
    local exit_code="$1"
    local job_name="$2"

    if [[ -z "${SLACK_WEBHOOK_URL:-}" ]]; then
        log_warn "NOTIFY_ON_FAILURE=1 but SLACK_WEBHOOK_URL is not set, skipping notification"
        return 0
    fi

    if ! command -v curl &>/dev/null; then
        log_warn "curl not found, skipping Slack notification"
        return 0
    fi

    local payload
    payload="$(cat <<-ENDJSON
{
  "text": ":red_circle: *CI Job Failed*",
  "blocks": [
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": ":red_circle: *CI Job Failed*\n*Job:* ${job_name}\n*Pipeline:* ${PIPELINE_ID}\n*Node:* ${SLURM_CI_NODELIST}\n*Exit code:* ${exit_code}\n*Git ref:* ${GIT_REF} (\`${GIT_SHORT_SHA}\`)"
      }
    }
  ]
}
ENDJSON
)"

    log_info "Sending Slack failure notification for job '${job_name}'"
    curl -sS -X POST \
        -H 'Content-type: application/json' \
        --data "$payload" \
        "$SLACK_WEBHOOK_URL" >/dev/null \
        || log_warn "Slack notification failed"
}

# ---------------------------------------------------------------------------
# register_cleanup_trap - Register cleanup_job as the EXIT trap
# ---------------------------------------------------------------------------
# This hooks into common.sh's cleanup handler system so it runs alongside
# any other registered cleanup handlers.
register_cleanup_trap() {
    register_cleanup 'cleanup_job $?'
    log_debug "Cleanup trap registered"
}
