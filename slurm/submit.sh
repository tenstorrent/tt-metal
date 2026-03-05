#!/usr/bin/env bash
# submit.sh - Universal launcher for Slurm CI workflows
#
# Usage:
#   ./slurm/submit.sh <workflow-name> [options] [-- extra-sbatch-args...]
#   ./slurm/submit.sh --list
#   ./slurm/submit.sh --status <pipeline-id>
#
# Examples:
#   ./slurm/submit.sh all-post-commit-workflows
#   ./slurm/submit.sh fast-dispatch-build-and-unit-tests --arch wormhole_b0
#   ./slurm/submit.sh galaxy-unit-tests --docker-image ghcr.io/.../dev:latest
#   ./slurm/submit.sh build-artifact --ref main -- --partition=build

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"
source_config env
export SLURM_CI_ARTIFACT_BASE="${ARTIFACT_BASE}"
source_lib artifacts

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<'USAGE'
Usage:
  ./slurm/submit.sh <workflow-name> [options] [-- extra-sbatch-args...]
  ./slurm/submit.sh --list
  ./slurm/submit.sh --status <pipeline-id>

Options:
  --docker-image IMAGE    Docker image override
  --arch ARCH             Architecture (wormhole_b0, blackhole, etc.)
  --ref REF               Git ref to build/test
  --pipeline-id ID        Override auto-generated pipeline ID
  --dry-run               Print what would be submitted without executing
  --list                  List available workflows
  --status PIPELINE_ID    Show status of all jobs in a pipeline
  --                      Pass remaining args directly to sbatch
USAGE
}

# ---------------------------------------------------------------------------
# list_workflows - Print names of available workflow scripts
# ---------------------------------------------------------------------------

list_workflows() {
    local workflows_dir="${SCRIPT_DIR}/workflows"
    echo "Available workflows:"
    local script
    for script in "${workflows_dir}"/*.sh; do
        [[ -f "$script" ]] || continue
        echo "  $(basename "$script" .sh)"
    done
}

# ---------------------------------------------------------------------------
# show_status - Display sacct information for a pipeline
# ---------------------------------------------------------------------------

show_status() {
    local pid="$1"
    local meta="${ARTIFACT_BASE}/${pid}/pipeline.json"

    if [[ -f "$meta" ]]; then
        log_info "Pipeline metadata (${meta}):"
        cat "$meta" >&2
        echo "" >&2
    else
        log_warn "No metadata found at ${meta}"
    fi

    require_cmd sacct

    if [[ -f "$meta" ]] && command -v jq &>/dev/null; then
        local -a job_ids
        mapfile -t job_ids < <(jq -r '.job_ids[]' "$meta" 2>/dev/null)
        if [[ ${#job_ids[@]} -gt 0 ]]; then
            local id
            for id in "${job_ids[@]}"; do
                sacct -j "$id" \
                    --format=JobID,JobName%40,State,ExitCode,Elapsed,Start,End \
                    --parsable2
            done
            return 0
        fi
    fi

    # Fallback: search sacct for jobs containing the pipeline ID in their name
    local start_date
    if date -u -d '7 days ago' '+%Y-%m-%dT%H:%M:%S' &>/dev/null 2>&1; then
        start_date="$(date -u -d '7 days ago' '+%Y-%m-%dT%H:%M:%S')"
    else
        start_date="$(date -u -v-7d '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || date -u '+%Y-%m-%dT%H:%M:%S')"
    fi

    sacct --starttime="${start_date}" \
        --format=JobID,JobName%40,State,ExitCode,Elapsed,Start,End \
        --parsable2 \
        -X | grep "${pid}" || log_warn "No jobs found matching pipeline ${pid}"
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

case "${1:-}" in
    --list)
        list_workflows
        exit 0
        ;;
    --status)
        [[ -z "${2:-}" ]] && log_fatal "Usage: $0 --status <pipeline-id>"
        show_status "$2"
        exit 0
        ;;
    --help|-h)
        usage
        exit 0
        ;;
    "")
        usage >&2
        exit 1
        ;;
esac

WORKFLOW_NAME="$1"
shift

OPT_DOCKER_IMAGE=""
OPT_ARCH=""
OPT_REF=""
OPT_PIPELINE_ID=""
DRY_RUN=0
EXTRA_SBATCH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --docker-image)  OPT_DOCKER_IMAGE="${2:?--docker-image requires a value}"; shift 2 ;;
        --arch)          OPT_ARCH="${2:?--arch requires a value}"; shift 2 ;;
        --ref)           OPT_REF="${2:?--ref requires a value}"; shift 2 ;;
        --pipeline-id)   OPT_PIPELINE_ID="${2:?--pipeline-id requires a value}"; shift 2 ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --)              shift; EXTRA_SBATCH_ARGS=("$@"); break ;;
        *)               log_fatal "Unknown option: $1 (use -- before extra sbatch args)" ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve workflow script
# ---------------------------------------------------------------------------

WORKFLOW_SCRIPT="${SCRIPT_DIR}/workflows/${WORKFLOW_NAME}.sh"

if [[ ! -f "${WORKFLOW_SCRIPT}" ]]; then
    log_error "Workflow not found: ${WORKFLOW_NAME}"
    log_error "Expected: ${WORKFLOW_SCRIPT}"
    echo ""
    list_workflows >&2
    exit 1
fi

if [[ ! -x "${WORKFLOW_SCRIPT}" ]]; then
    log_warn "Workflow script is not executable — adding +x: ${WORKFLOW_SCRIPT}"
    chmod +x "${WORKFLOW_SCRIPT}"
fi

# ---------------------------------------------------------------------------
# Set environment from flags (flags override env vars)
# ---------------------------------------------------------------------------

[[ -n "$OPT_PIPELINE_ID" ]] && PIPELINE_ID="$OPT_PIPELINE_ID"
[[ -n "$OPT_DOCKER_IMAGE" ]] && DOCKER_IMAGE="$OPT_DOCKER_IMAGE"
[[ -n "$OPT_ARCH" ]] && ARCH_NAME="$OPT_ARCH"
[[ -n "$OPT_REF" ]] && GIT_REF="$OPT_REF"

export PIPELINE_ID DOCKER_IMAGE="${DOCKER_IMAGE:-}" ARCH_NAME="${ARCH_NAME:-}" GIT_REF

ARTIFACT_DIR="$(get_artifact_dir "$PIPELINE_ID")"
LOG_DIR="${LOG_BASE}/${PIPELINE_ID}"
export ARTIFACT_DIR LOG_DIR

# ---------------------------------------------------------------------------
# Create artifact and log directories on Weka
# ---------------------------------------------------------------------------

mkdir -p "${ARTIFACT_DIR}" "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

log_info "=============================================="
log_info "  Slurm CI Pipeline Launcher"
log_info "=============================================="
log_info "  Workflow:    ${WORKFLOW_NAME}"
log_info "  Pipeline ID: ${PIPELINE_ID}"
log_info "  Git SHA:     ${GIT_SHORT_SHA}"
log_info "  Git Ref:     ${GIT_REF}"
log_info "  Arch:        ${ARCH_NAME:-default}"
log_info "  Docker:      ${DOCKER_IMAGE:-<to be resolved>}"
log_info "  Artifacts:   ${ARTIFACT_DIR}"
log_info "  Logs:        ${LOG_DIR}"
log_info "=============================================="

# ---------------------------------------------------------------------------
# Dry-run guard
# ---------------------------------------------------------------------------

if [[ $DRY_RUN -eq 1 ]]; then
    log_info "[DRY RUN] Would submit:"
    log_info "  sbatch --parsable \\"
    log_info "    --job-name=ci-${WORKFLOW_NAME}-${PIPELINE_ID} \\"
    log_info "    --output=${LOG_DIR}/%x-%j.out \\"
    log_info "    --error=${LOG_DIR}/%x-%j.err \\"
    if [[ ${#EXTRA_SBATCH_ARGS[@]} -gt 0 ]]; then
        log_info "    ${EXTRA_SBATCH_ARGS[*]} \\"
    fi
    log_info "    ${WORKFLOW_SCRIPT}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Submit via sbatch
# ---------------------------------------------------------------------------

require_cmd sbatch

SBATCH_CMD=(
    sbatch
    --parsable
    "--job-name=ci-${WORKFLOW_NAME}-${PIPELINE_ID}"
    "--output=${LOG_DIR}/%x-%j.out"
    "--error=${LOG_DIR}/%x-%j.err"
    "--export=ALL,PIPELINE_ID=${PIPELINE_ID},ARTIFACT_DIR=${ARTIFACT_DIR},LOG_DIR=${LOG_DIR}"
)

if [[ ${#EXTRA_SBATCH_ARGS[@]} -gt 0 ]]; then
    SBATCH_CMD+=("${EXTRA_SBATCH_ARGS[@]}")
fi

SBATCH_CMD+=("${WORKFLOW_SCRIPT}")

SBATCH_OUTPUT="$("${SBATCH_CMD[@]}")"
JOBID="${SBATCH_OUTPUT%%_*}"

if [[ -z "${JOBID}" ]]; then
    log_fatal "sbatch returned empty job ID — submission failed"
fi

log_info "Submitted job ${JOBID} for workflow '${WORKFLOW_NAME}'"
echo "${JOBID}"

# ---------------------------------------------------------------------------
# Save pipeline metadata as JSON
# ---------------------------------------------------------------------------

require_cmd jq

jq -n \
    --arg pid "$PIPELINE_ID" \
    --arg wf "$WORKFLOW_NAME" \
    --arg sha "$GIT_SHA" \
    --arg ref "$GIT_REF" \
    --arg arch "${ARCH_NAME:-}" \
    --arg img "${DOCKER_IMAGE:-}" \
    --arg ts "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
    --arg by "$(whoami)@$(hostname)" \
    --arg jid "$JOBID" \
    '{
        pipeline_id: $pid,
        workflow: $wf,
        git_sha: $sha,
        git_ref: $ref,
        arch: $arch,
        docker_image: $img,
        submitted_at: $ts,
        submitted_by: $by,
        job_ids: [$jid]
    }' > "${ARTIFACT_DIR}/pipeline.json"

log_info "Pipeline metadata saved to ${ARTIFACT_DIR}/pipeline.json"
