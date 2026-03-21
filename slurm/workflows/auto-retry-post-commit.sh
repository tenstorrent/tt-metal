#!/usr/bin/env bash
#SBATCH --job-name=auto-retry-post-commit
#SBATCH --partition=build
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#
# Check sacct for failed post-commit jobs and resubmit with --requeue.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib retry
source_config env

require_cmd sacct
require_cmd squeue
require_cmd scontrol

MAX_RETRIES="${MAX_RETRIES:-1}"
LOOKBACK_HOURS="${LOOKBACK_HOURS:-4}"
TARGET_PREFIX="${TARGET_PREFIX:-all-post-commit}"

log_info "=== Auto-retry post-commit starting ==="
log_info "  Lookback: ${LOOKBACK_HOURS}h"
log_info "  Max retries: ${MAX_RETRIES}"
log_info "  Target prefix: ${TARGET_PREFIX}"

START_TIME="$(date -u -d "${LOOKBACK_HOURS} hours ago" '+%Y-%m-%dT%H:%M' 2>/dev/null || \
    date -u -v-${LOOKBACK_HOURS}H '+%Y-%m-%dT%H:%M')"

FAILED_JOBS="$(sacct \
    --starttime="${START_TIME}" \
    --format=JobID,JobName%50,State,ExitCode \
    --noheader \
    --parsable2 \
    --state=FAILED,TIMEOUT,NODE_FAIL \
    | grep "${TARGET_PREFIX}" || true)"

if [[ -z "${FAILED_JOBS}" ]]; then
    log_info "No failed post-commit jobs found in the last ${LOOKBACK_HOURS} hours"
    exit 0
fi

RETRY_COUNT=0
SKIP_COUNT=0

while IFS='|' read -r job_id job_name state exit_code; do
    [[ -z "${job_id}" ]] && continue
    base_id="${job_id%%_*}"

    EXISTING_RETRIES="$(sacct \
        --name="${job_name}" \
        --starttime="${START_TIME}" \
        --format=State \
        --noheader \
        --parsable2 \
        | grep -c 'PENDING\|RUNNING' || echo 0)"

    if (( EXISTING_RETRIES >= MAX_RETRIES )); then
        log_info "Skipping ${job_name} (${base_id}): already ${EXISTING_RETRIES} retries pending/running"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        continue
    fi

    log_info "Requeuing ${job_name} (${base_id}, was ${state})"
    if scontrol requeue "${base_id}" 2>/dev/null; then
        RETRY_COUNT=$((RETRY_COUNT + 1))
        log_info "  Requeued successfully"
    else
        log_warn "  Requeue failed for ${base_id}, job may have been cleaned up"
    fi
done <<< "${FAILED_JOBS}"

log_info "=== Auto-retry post-commit complete ==="
log_info "  Retried: ${RETRY_COUNT}"
log_info "  Skipped: ${SKIP_COUNT}"
