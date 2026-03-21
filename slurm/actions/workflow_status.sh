#!/usr/bin/env bash
# workflow_status.sh - Check Slurm job statuses via sacct for a pipeline.
# Equivalent to .github/actions/workflow-status/action.yml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") --pipeline-id ID [OPTIONS]

Query sacct for all jobs belonging to a pipeline and verify their status.
Exits 0 if all required jobs succeeded, 1 otherwise.

Required:
  --pipeline-id ID              Pipeline identifier to query

Options:
  --required-jobs JOB1,JOB2     Comma-separated list of required jobs (must succeed)
  --optional-jobs JOB1,JOB2     Comma-separated list of optional jobs (skipped OK, failure not)
  -h, --help                    Show this help message

Environment:
  PIPELINE_ID                   Fallback for --pipeline-id
EOF
    exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

QUERY_PIPELINE="${PIPELINE_ID:-}"
REQUIRED_JOBS=""
OPTIONAL_JOBS=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pipeline-id)    QUERY_PIPELINE="$2"; shift 2 ;;
        --required-jobs)  REQUIRED_JOBS="$2"; shift 2 ;;
        --optional-jobs)  OPTIONAL_JOBS="$2"; shift 2 ;;
        -h|--help)        usage 0 ;;
        *)                log_error "Unknown option: $1"; usage 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ -z "${QUERY_PIPELINE}" ]]; then
    log_error "--pipeline-id is required"
    usage 1
fi

require_cmd sacct

# ---------------------------------------------------------------------------
# Query sacct
# ---------------------------------------------------------------------------

log_info "Querying job status for pipeline: ${QUERY_PIPELINE}"

SACCT_FORMAT="JobID,JobName%40,Partition,State,ExitCode,Elapsed,Start,End"

declare -A JOB_STATES=()

while IFS='|' read -r jobid name partition state exitcode elapsed start end; do
    [[ -z "${jobid}" ]] && continue
    name="${name## }"; name="${name%% }"
    printf "%-12s %-40s %-12s %-12s %-8s %s\n" \
        "${jobid}" "${name}" "${partition}" "${state}" "${exitcode}" "${elapsed}"
    JOB_STATES["${name}"]="${state}"
done < <(sacct --name="*${QUERY_PIPELINE}*" \
    --format="${SACCT_FORMAT}" \
    --noheader \
    --parsable2 2>/dev/null)

echo ""

# ---------------------------------------------------------------------------
# Check required jobs
# ---------------------------------------------------------------------------

FAILURES=0

if [[ -n "${REQUIRED_JOBS}" ]]; then
    IFS=',' read -ra REQ_LIST <<< "${REQUIRED_JOBS}"
    for job in "${REQ_LIST[@]}"; do
        job="${job## }"; job="${job%% }"
        state="${JOB_STATES["${job}"]:-UNKNOWN}"
        if [[ "${state}" != "COMPLETED" ]]; then
            log_error "Required job '${job}' did not succeed (state: ${state})"
            FAILURES=$((FAILURES + 1))
        else
            log_info "Required job '${job}' succeeded"
        fi
    done
fi

# ---------------------------------------------------------------------------
# Check optional jobs
# ---------------------------------------------------------------------------

if [[ -n "${OPTIONAL_JOBS}" ]]; then
    IFS=',' read -ra OPT_LIST <<< "${OPTIONAL_JOBS}"
    for job in "${OPT_LIST[@]}"; do
        job="${job## }"; job="${job%% }"
        state="${JOB_STATES["${job}"]:-SKIPPED}"
        case "${state}" in
            COMPLETED|SKIPPED|UNKNOWN) ;;
            FAILED|CANCELLED)
                log_error "Optional job '${job}' failed (state: ${state})"
                FAILURES=$((FAILURES + 1))
                ;;
            *)
                log_warn "Optional job '${job}' in state: ${state}"
                ;;
        esac
    done
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

TOTAL="${#JOB_STATES[@]}"
COMPLETED=0
FAILED_COUNT=0
RUNNING=0

for state in "${JOB_STATES[@]}"; do
    case "${state}" in
        COMPLETED)  COMPLETED=$((COMPLETED + 1)) ;;
        FAILED)     FAILED_COUNT=$((FAILED_COUNT + 1)) ;;
        RUNNING)    RUNNING=$((RUNNING + 1)) ;;
    esac
done

echo "Summary: total=${TOTAL} completed=${COMPLETED} running=${RUNNING} failed=${FAILED_COUNT}"

if (( FAILURES > 0 )); then
    log_error "${FAILURES} job check(s) failed"
    exit 1
fi

log_info "All job checks passed"
