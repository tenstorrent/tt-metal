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
    cat <<'HELP'
submit.sh - Universal launcher for Slurm CI workflows

USAGE
  ./slurm/submit.sh <workflow-name> [options] [-- extra-sbatch-args...]
  ./slurm/submit.sh --list
  ./slurm/submit.sh --status <pipeline-id>
  ./slurm/submit.sh --help

COMMANDS
  <workflow-name>           Submit the named workflow as a Slurm job.
                            Workflows live in slurm/workflows/<name>.sh.
  --list                    List all available workflow names and exit.
  --status <pipeline-id>    Show sacct status for every job in a pipeline.

OPTIONS
  --partition PARTITION      Slurm partition to submit the job to. Overrides
                            any #SBATCH --partition directive in the workflow
                            script. If omitted and scontrol reports exactly
                            one UP partition, that partition is used
                            automatically.
  --docker-image IMAGE      Override the Docker image used by the workflow.
                            Bypasses image resolution and Harbor mirror logic.
  --arch ARCH               Target architecture passed to the job as ARCH_NAME.
                            Common values: wormhole_b0, blackhole.
  --ref REF                 Git ref (branch, tag, or SHA) to build/test.
                            Overrides the auto-detected GIT_REF.
  --pipeline-id ID          Set an explicit pipeline ID instead of the default
                            auto-generated <timestamp>-<short-sha> format.
  --dry-run                 Print the sbatch command that would be executed
                            without actually submitting the job.
  -h, --help                Show this help message and exit.
  --                        Stop option parsing. Everything after '--' is
                            forwarded verbatim as extra sbatch arguments
                            (e.g. --time=01:00:00).

ENVIRONMENT VARIABLES
  The following variables can be exported before invocation to customise
  behaviour. CLI flags take precedence over environment variables.

  CI_STORAGE_BASE           Root of shared CI storage. Artifacts and logs are
                            stored beneath this path.
                            Default: $(pwd)/.slurm-ci
  PIPELINE_ID               Override the auto-generated pipeline identifier.
  DOCKER_IMAGE              Docker image for the workflow (same as --docker-image).
  ARCH_NAME                 Target architecture (same as --arch).
  GIT_REF                   Git ref to build/test (same as --ref).
  MLPERF_BASE               Mount point for MLPerf / model data.
                            Default: /mnt/MLPerf
  CONTAINER_WORKDIR         Working directory inside Docker containers.
                            Default: /work
  SLACK_WEBHOOK_URL         Slack webhook for failure notifications (optional).

STORAGE LAYOUT
  All derived paths are rooted at CI_STORAGE_BASE (see config/site.sh):

    ${CI_STORAGE_BASE}/
    ├── artifacts/<pipeline-id>/       Build tarballs, wheels, reports
    │   ├── build/ttm_any.tar.zst
    │   ├── docker/image_tags.env
    │   ├── reports/<job-name>/
    │   └── pipeline.json
    └── logs/<pipeline-id>/            Job stdout/stderr logs
        └── <job-name>-<job-id>.out

EXAMPLES
  # Run the full post-commit pipeline
  ./slurm/submit.sh all-post-commit-workflows

  # Run a specific test suite on Blackhole hardware
  ./slurm/submit.sh fast-dispatch-build-and-unit-tests --arch blackhole

  # Use a specific Docker image
  ./slurm/submit.sh galaxy-unit-tests \
      --docker-image ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest

  # Build only, targeting the main branch, on the build partition
  ./slurm/submit.sh build-artifact --ref main -- --partition=build

  # Override the partition (ignores workflow #SBATCH --partition)
  ./slurm/submit.sh multi-host-physical --partition debug

  # Dry-run to see what would be submitted
  ./slurm/submit.sh multi-host-physical --dry-run

  # Point CI storage at a dedicated NFS path
  export CI_STORAGE_BASE=/data/ci
  ./slurm/submit.sh smoke

  # Check pipeline status
  ./slurm/submit.sh --status 20260310-143022-a1b2c3d

  # List all available workflows
  ./slurm/submit.sh --list
HELP
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
OPT_PARTITION=""
DRY_RUN=0
EXTRA_SBATCH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition)     OPT_PARTITION="${2:?--partition requires a value}"; shift 2 ;;
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
# Partition resolution
# ---------------------------------------------------------------------------

resolve_partition() {
    # 1. Explicit --partition flag
    if [[ -n "${OPT_PARTITION}" ]]; then
        echo "$OPT_PARTITION"
        return 0
    fi

    # 2. SLURM_PARTITION env var (without the --partition= prefix if present)
    if [[ -n "${SLURM_PARTITION:-}" ]]; then
        echo "${SLURM_PARTITION#--partition=}"
        return 0
    fi

    # 3. Auto-detect from scontrol
    if ! command -v scontrol &>/dev/null; then
        return 1
    fi

    local -a up_partitions=()
    local line
    while IFS= read -r line; do
        up_partitions+=("$line")
    done < <(scontrol show partitions 2>/dev/null \
        | awk '/^PartitionName=/ { name=$1; sub(/PartitionName=/,"",name) }
               /State=UP/        { print name }')

    if [[ ${#up_partitions[@]} -eq 0 ]]; then
        log_warn "No UP partitions found via scontrol"
        return 1
    fi

    if [[ ${#up_partitions[@]} -eq 1 ]]; then
        log_info "Auto-detected single UP partition: ${up_partitions[0]}"
        echo "${up_partitions[0]}"
        return 0
    fi

    log_warn "Multiple partitions available (${up_partitions[*]}); use --partition to select one"
    return 1
}

RESOLVED_PARTITION=""
if resolved="$(resolve_partition)"; then
    RESOLVED_PARTITION="$resolved"
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
# Create artifact and log directories
# ---------------------------------------------------------------------------

mkdir -p "${ARTIFACT_DIR}" "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Generate wrapper script for sbatch dispatch
# ---------------------------------------------------------------------------
# sbatch copies the submitted script to a staging area on the worker node
# (e.g. /var/lib/slurm/slurmctld/job<id>/slurm_script).  The workflow scripts
# use BASH_SOURCE[0] to locate the slurm/ library tree, which breaks when the
# path resolves to the slurmctld copy instead of the NFS original.
#
# The wrapper preserves the workflow's #SBATCH directives (so sbatch honours
# --time, --job-name, etc.) then exec's the real script from its NFS path,
# giving BASH_SOURCE[0] the correct value.

WRAPPER_SCRIPT="${LOG_DIR}/wrapper-${WORKFLOW_NAME}.sh"
{
    echo "#!/usr/bin/env bash"
    # Preserve #SBATCH directives from the workflow so sbatch applies them
    awk '/^#SBATCH/ { print }' "${WORKFLOW_SCRIPT}"
    echo "export SLURM_SCRIPTS_DIR='${SLURM_SCRIPTS_DIR}'"
    echo "exec bash '${WORKFLOW_SCRIPT}' \"\$@\""
} > "${WRAPPER_SCRIPT}"
chmod +x "${WRAPPER_SCRIPT}"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

log_info "=============================================="
log_info "  Slurm CI Pipeline Launcher"
log_info "=============================================="
log_info "  Workflow:    ${WORKFLOW_NAME}"
log_info "  Pipeline ID: ${PIPELINE_ID}"
log_info "  Partition:   ${RESOLVED_PARTITION:-<from workflow #SBATCH>}"
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
    log_info "[DRY RUN] Would submit wrapper: ${WRAPPER_SCRIPT}"
    log_info "  Wrapper exec's: ${WORKFLOW_SCRIPT}"
    log_info ""
    log_info "  sbatch --parsable \\"
    log_info "    --job-name=ci-${WORKFLOW_NAME}-${PIPELINE_ID} \\"
    if [[ -n "${RESOLVED_PARTITION}" ]]; then
        log_info "    --partition=${RESOLVED_PARTITION} \\"
    fi
    log_info "    --output=${LOG_DIR}/%x-%j.out \\"
    log_info "    --error=${LOG_DIR}/%x-%j.err \\"
    if [[ ${#EXTRA_SBATCH_ARGS[@]} -gt 0 ]]; then
        log_info "    ${EXTRA_SBATCH_ARGS[*]} \\"
    fi
    log_info "    ${WRAPPER_SCRIPT}"
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
    "--export=ALL,PIPELINE_ID=${PIPELINE_ID},ARTIFACT_DIR=${ARTIFACT_DIR},LOG_DIR=${LOG_DIR},SLURM_SCRIPTS_DIR=${SLURM_SCRIPTS_DIR}"
)

if [[ -n "${RESOLVED_PARTITION}" ]]; then
    SBATCH_CMD+=("--partition=${RESOLVED_PARTITION}")
fi

if [[ ${#EXTRA_SBATCH_ARGS[@]} -gt 0 ]]; then
    SBATCH_CMD+=("${EXTRA_SBATCH_ARGS[@]}")
fi

SBATCH_CMD+=("${WRAPPER_SCRIPT}")

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
    --arg partition "${RESOLVED_PARTITION:-}" \
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
        partition: $partition,
        docker_image: $img,
        submitted_at: $ts,
        submitted_by: $by,
        job_ids: [$jid]
    }' > "${ARTIFACT_DIR}/pipeline.json"

log_info "Pipeline metadata saved to ${ARTIFACT_DIR}/pipeline.json"
