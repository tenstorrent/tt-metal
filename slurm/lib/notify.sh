#!/usr/bin/env bash
# notify.sh - Slack notification helpers for Slurm CI
# Port of .github/actions/slack-report and related notification patterns.
# Requires: curl, SLACK_WEBHOOK_URL or SLACK_BOT_TOKEN env var.

set -euo pipefail

[[ -n "${_SLURM_CI_NOTIFY_SH:-}" ]] && return 0
_SLURM_CI_NOTIFY_SH=1

# ---------------------------------------------------------------------------
# send_slack_message - POST a simple text payload to a Slack webhook
# ---------------------------------------------------------------------------
# Usage: send_slack_message <webhook_url> <message>
send_slack_message() {
    local webhook_url="${1:-${SLACK_WEBHOOK_URL:-}}"
    local message="$2"

    if [[ -z "${webhook_url}" ]]; then
        log_warn "No Slack webhook URL provided; skipping notification"
        return 0
    fi

    local payload
    payload=$(printf '{"text": %s}' "$(printf '%s' "${message}" | jq -Rs .)")

    local http_code
    http_code=$(curl -sS -o /dev/null -w '%{http_code}' \
        -X POST \
        -H 'Content-Type: application/json' \
        -d "${payload}" \
        "${webhook_url}") || {
        log_warn "curl failed while sending Slack message"
        return 1
    }

    if [[ "${http_code}" -lt 200 || "${http_code}" -ge 300 ]]; then
        log_warn "Slack webhook returned HTTP ${http_code}"
        return 1
    fi
    log_info "Slack message sent (HTTP ${http_code})"
}

# ---------------------------------------------------------------------------
# send_slack_block - POST a blocks-format message to a Slack webhook
# ---------------------------------------------------------------------------
# Usage: send_slack_block <webhook_url> <blocks_json>
#   blocks_json should be a valid JSON array of Slack Block Kit blocks.
send_slack_block() {
    local webhook_url="${1:-${SLACK_WEBHOOK_URL:-}}"
    local blocks_json="$2"

    if [[ -z "${webhook_url}" ]]; then
        log_warn "No Slack webhook URL provided; skipping notification"
        return 0
    fi

    local payload
    payload=$(printf '{"blocks": %s}' "${blocks_json}")

    local http_code
    http_code=$(curl -sS -o /dev/null -w '%{http_code}' \
        -X POST \
        -H 'Content-Type: application/json' \
        -d "${payload}" \
        "${webhook_url}") || {
        log_warn "curl failed while sending Slack block message"
        return 1
    }

    if [[ "${http_code}" -lt 200 || "${http_code}" -ge 300 ]]; then
        log_warn "Slack webhook returned HTTP ${http_code}"
        return 1
    fi
    log_info "Slack block message sent (HTTP ${http_code})"
}

# ---------------------------------------------------------------------------
# format_failure_report - Build a Slack blocks payload for a failed job
# ---------------------------------------------------------------------------
# Usage: format_failure_report <job_name> <exit_code> <log_file>
#   Includes the last 20 lines of log_file in the notification.
#   Outputs JSON to stdout.
format_failure_report() {
    local job_name="$1"
    local exit_code="$2"
    local log_file="${3:-}"

    local log_snippet=""
    if [[ -n "${log_file}" && -f "${log_file}" ]]; then
        log_snippet=$(tail -20 "${log_file}" | sed 's/"/\\"/g' | sed 's/$/\\n/' | tr -d '\n')
    fi

    cat <<EOJSON
{
  "blocks": [
    {
      "type": "header",
      "text": {"type": "plain_text", "text": ":x: Slurm CI Job Failed"}
    },
    {
      "type": "section",
      "fields": [
        {"type": "mrkdwn", "text": "*Job:* ${job_name}"},
        {"type": "mrkdwn", "text": "*Exit Code:* ${exit_code}"},
        {"type": "mrkdwn", "text": "*Pipeline:* ${PIPELINE_ID:-unknown}"},
        {"type": "mrkdwn", "text": "*Ref:* ${GIT_REF:-unknown}"},
        {"type": "mrkdwn", "text": "*SHA:* ${GIT_SHORT_SHA:-unknown}"}
      ]
    }${log_snippet:+,
    {
      "type": "section",
      "text": {"type": "mrkdwn", "text": "*Last 20 lines of log:*\n\`\`\`${log_snippet}\`\`\`"}
    }}
  ]
}
EOJSON
}

# ---------------------------------------------------------------------------
# format_success_report - Build a Slack blocks payload for a successful job
# ---------------------------------------------------------------------------
# Usage: format_success_report <job_name> <elapsed_seconds>
#   Outputs JSON to stdout.
format_success_report() {
    local job_name="$1"
    local elapsed_seconds="$2"

    local minutes seconds duration_str
    minutes=$((elapsed_seconds / 60))
    seconds=$((elapsed_seconds % 60))
    duration_str="${minutes}m ${seconds}s"

    cat <<EOJSON
{
  "blocks": [
    {
      "type": "header",
      "text": {"type": "plain_text", "text": ":white_check_mark: Slurm CI Job Succeeded"}
    },
    {
      "type": "section",
      "fields": [
        {"type": "mrkdwn", "text": "*Job:* ${job_name}"},
        {"type": "mrkdwn", "text": "*Duration:* ${duration_str}"},
        {"type": "mrkdwn", "text": "*Pipeline:* ${PIPELINE_ID:-unknown}"},
        {"type": "mrkdwn", "text": "*Ref:* ${GIT_REF:-unknown}"},
        {"type": "mrkdwn", "text": "*SHA:* ${GIT_SHORT_SHA:-unknown}"}
      ]
    }
  ]
}
EOJSON
}

# ---------------------------------------------------------------------------
# send_pipeline_summary - Read job results file and send aggregate report
# ---------------------------------------------------------------------------
# Usage: send_pipeline_summary <webhook_url> <pipeline_id> <job_results_file>
#   job_results_file contains lines of "job_name:exit_code".
send_pipeline_summary() {
    local webhook_url="${1:-${SLACK_WEBHOOK_URL:-}}"
    local pipeline_id="${2:-${PIPELINE_ID:-unknown}}"
    local job_results_file="$3"

    if [[ ! -f "${job_results_file}" ]]; then
        log_warn "Job results file not found: ${job_results_file}"
        return 1
    fi

    local total=0 passed=0 failed=0
    local failed_jobs=""

    while IFS=: read -r job_name exit_code; do
        [[ -z "${job_name}" ]] && continue
        total=$((total + 1))
        if [[ "${exit_code}" -eq 0 ]]; then
            passed=$((passed + 1))
        else
            failed=$((failed + 1))
            failed_jobs="${failed_jobs}• ${job_name} (exit ${exit_code})\n"
        fi
    done < "${job_results_file}"

    local status_emoji=":white_check_mark:"
    local status_text="All ${total} jobs passed"
    if [[ ${failed} -gt 0 ]]; then
        status_emoji=":x:"
        status_text="${passed}/${total} passed, ${failed} failed"
    fi

    local failed_section=""
    if [[ -n "${failed_jobs}" ]]; then
        failed_section=',
    {
      "type": "section",
      "text": {"type": "mrkdwn", "text": "*Failed jobs:*\n'"${failed_jobs}"'"}
    }'
    fi

    local payload
    payload=$(cat <<EOJSON
{
  "blocks": [
    {
      "type": "header",
      "text": {"type": "plain_text", "text": "${status_emoji} Pipeline Summary: ${pipeline_id}"}
    },
    {
      "type": "section",
      "fields": [
        {"type": "mrkdwn", "text": "*Status:* ${status_text}"},
        {"type": "mrkdwn", "text": "*Total:* ${total}"},
        {"type": "mrkdwn", "text": "*Ref:* ${GIT_REF:-unknown}"},
        {"type": "mrkdwn", "text": "*SHA:* ${GIT_SHORT_SHA:-unknown}"}
      ]
    }${failed_section}
  ]
}
EOJSON
)
    send_slack_block "${webhook_url}" "$(echo "${payload}" | jq -c '.blocks')"
}

# ---------------------------------------------------------------------------
# notify_on_failure - Send notification only when exit_code != 0
# ---------------------------------------------------------------------------
# Usage: notify_on_failure <webhook_url> <job_name> <exit_code> <log_file>
notify_on_failure() {
    local webhook_url="${1:-${SLACK_WEBHOOK_URL:-}}"
    local job_name="$2"
    local exit_code="$3"
    local log_file="${4:-}"

    if [[ "${exit_code}" -eq 0 ]]; then
        return 0
    fi

    local payload
    payload=$(format_failure_report "${job_name}" "${exit_code}" "${log_file}")
    send_slack_block "${webhook_url}" "$(echo "${payload}" | jq -c '.blocks')"
}
