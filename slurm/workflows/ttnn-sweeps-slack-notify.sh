#!/usr/bin/env bash
#SBATCH --job-name=ttnn-sweeps-slack-notify
#SBATCH --partition=build
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#
# Sweep results Slack notification. Collects test reports from the pipeline
# and posts a summary to Slack.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib notify
source_config env

require_env PIPELINE_ID

NOTIFY_TYPE="${NOTIFY_TYPE:-sweep}"

log_info "=== TTNN sweeps Slack notify starting ==="
log_info "  Pipeline:    ${PIPELINE_ID}"
log_info "  Notify type: ${NOTIFY_TYPE}"

REPORT_DIR="${ARTIFACT_DIR}/reports"

if [[ ! -d "${REPORT_DIR}" ]]; then
    log_warn "No report directory found at ${REPORT_DIR}, skipping notification"
    exit 0
fi

TOTAL=0
PASSED=0
FAILED=0
FAILED_JOBS=""

for job_dir in "${REPORT_DIR}"/*/; do
    [[ -d "${job_dir}" ]] || continue
    job_name="$(basename "${job_dir}")"
    TOTAL=$((TOTAL + 1))
    if [[ -f "${job_dir}/FAILED" ]]; then
        FAILED=$((FAILED + 1))
        FAILED_JOBS="${FAILED_JOBS}\n- ${job_name}"
    else
        PASSED=$((PASSED + 1))
    fi
done

log_info "Results: ${PASSED}/${TOTAL} passed, ${FAILED} failed"

send_pipeline_summary "${PIPELINE_ID}" "${SLACK_WEBHOOK_URL:-}"

if [[ ${FAILED} -gt 0 && -n "${SLACK_WEBHOOK_URL:-}" ]]; then
    DETAIL_PAYLOAD=$(cat <<EOJSON
{
  "blocks": [
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": ":warning: *Failed jobs (${NOTIFY_TYPE}):*$(echo -e "${FAILED_JOBS}")"
      }
    }
  ]
}
EOJSON
)
    send_slack_message "${SLACK_WEBHOOK_URL}" "${DETAIL_PAYLOAD}"
fi

log_info "=== TTNN sweeps Slack notify complete ==="
