#!/usr/bin/env bash
# matrix.sh - Job array and matrix strategy support
# Maps GitHub Actions matrix strategies to Slurm job arrays using JSON + jq.

set -euo pipefail

# Guard against double-sourcing
[[ -n "${_SLURM_CI_MATRIX_SH:-}" ]] && return 0
_SLURM_CI_MATRIX_SH=1

SLURM_CI_LIB_DIR="${SLURM_CI_LIB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SLURM_CI_LIB_DIR}/common.sh"

MATRIX_DIR="${MATRIX_DIR:-/tmp/slurm-matrix-${PIPELINE_ID}}"
mkdir -p "$MATRIX_DIR"

# ---------------------------------------------------------------------------
# create_matrix_file - Write matrix JSON array to a file
# ---------------------------------------------------------------------------
# Usage: create_matrix_file '<json_array>'
# Returns: path to the created matrix file
create_matrix_file() {
    local json_array="${1:?json_array required}"
    require_cmd jq

    # Validate input is a JSON array
    if ! echo "$json_array" | jq -e 'type == "array"' >/dev/null 2>&1; then
        log_error "create_matrix_file: input is not a valid JSON array"
        return 1
    fi

    local matrix_file="${MATRIX_DIR}/matrix-$(date -u '+%s')-$$.json"
    echo "$json_array" | jq '.' > "$matrix_file"
    log_info "Matrix file created: ${matrix_file} ($(get_matrix_size "$matrix_file") tasks)"
    echo "$matrix_file"
}

# ---------------------------------------------------------------------------
# get_matrix_size - Return the number of elements in the matrix
# ---------------------------------------------------------------------------
# Usage: get_matrix_size <matrix_file>
get_matrix_size() {
    local matrix_file="${1:?matrix_file required}"
    require_cmd jq

    jq 'length' "$matrix_file"
}

# ---------------------------------------------------------------------------
# get_task_config - Return the JSON object for a given task ID
# ---------------------------------------------------------------------------
# Usage: get_task_config <matrix_file> <task_id>
get_task_config() {
    local matrix_file="${1:?matrix_file required}"
    local task_id="${2:?task_id required}"
    require_cmd jq

    local size
    size="$(get_matrix_size "$matrix_file")"
    if (( task_id < 0 || task_id >= size )); then
        log_error "Task ID ${task_id} out of range [0, $((size - 1)))]"
        return 1
    fi

    jq ".[$task_id]" "$matrix_file"
}

# ---------------------------------------------------------------------------
# get_task_field - Return a specific field from a task config
# ---------------------------------------------------------------------------
# Usage: get_task_field <matrix_file> <task_id> <field>
get_task_field() {
    local matrix_file="${1:?matrix_file required}"
    local task_id="${2:?task_id required}"
    local field="${3:?field required}"
    require_cmd jq

    jq -r ".[$task_id].$field // empty" "$matrix_file"
}

# ---------------------------------------------------------------------------
# submit_job_array - Submit an sbatch job array for the matrix
# ---------------------------------------------------------------------------
# Usage: submit_job_array <script> <matrix_file> [extra_sbatch_args...]
# Returns: the Slurm JOBID
submit_job_array() {
    local script="${1:?script required}"; shift
    local matrix_file="${1:?matrix_file required}"; shift
    local -a extra_args=("$@")

    require_cmd sbatch

    local size
    size="$(get_matrix_size "$matrix_file")"
    if (( size == 0 )); then
        log_error "Matrix is empty, nothing to submit"
        return 1
    fi

    local array_spec="0-$((size - 1))"
    log_info "Submitting job array: ${script} (${size} tasks, array=${array_spec})"

    local output
    output="$(sbatch \
        --parsable \
        --array="$array_spec" \
        --export="ALL,MATRIX_FILE=${matrix_file}" \
        "${extra_args[@]}" \
        "$script")"

    local jobid="${output%%_*}"
    log_info "Job array submitted: JOBID=${jobid}"
    echo "$jobid"
}

# ---------------------------------------------------------------------------
# submit_dependent_job - Submit a job that depends on successful completion
# ---------------------------------------------------------------------------
# Usage: submit_dependent_job <script> <dependency_jobid> [extra_args...]
# Returns: the Slurm JOBID
submit_dependent_job() {
    local script="${1:?script required}"; shift
    local dependency_jobid="${1:?dependency_jobid required}"; shift
    local -a extra_args=("$@")

    require_cmd sbatch

    log_info "Submitting dependent job (afterok:${dependency_jobid}): ${script}"

    local output
    output="$(sbatch \
        --parsable \
        --dependency="afterok:${dependency_jobid}" \
        "${extra_args[@]}" \
        "$script")"

    local jobid="${output%%_*}"
    log_info "Dependent job submitted: JOBID=${jobid} (depends on ${dependency_jobid})"
    echo "$jobid"
}

# ---------------------------------------------------------------------------
# submit_dependent_job_any - Submit a job that runs after completion (any exit)
# ---------------------------------------------------------------------------
# Usage: submit_dependent_job_any <script> <dependency_jobid> [extra_args...]
# Returns: the Slurm JOBID
submit_dependent_job_any() {
    local script="${1:?script required}"; shift
    local dependency_jobid="${1:?dependency_jobid required}"; shift
    local -a extra_args=("$@")

    require_cmd sbatch

    log_info "Submitting dependent job (afterany:${dependency_jobid}): ${script}"

    local output
    output="$(sbatch \
        --parsable \
        --dependency="afterany:${dependency_jobid}" \
        "${extra_args[@]}" \
        "$script")"

    local jobid="${output%%_*}"
    log_info "Dependent job submitted: JOBID=${jobid} (depends-any on ${dependency_jobid})"
    echo "$jobid"
}

# ---------------------------------------------------------------------------
# wait_for_job - Poll until a job completes, return its exit code
# ---------------------------------------------------------------------------
# Usage: wait_for_job <jobid>
# Returns: exit code of the job (0 = success)
wait_for_job() {
    local jobid="${1:?jobid required}"
    local poll_interval=10
    local max_poll_interval=60

    require_cmd squeue sacct

    log_info "Waiting for job ${jobid} to complete..."

    while squeue -j "$jobid" -h --noheader 2>/dev/null | grep -q .; do
        sleep "$poll_interval"
        if (( poll_interval < max_poll_interval )); then
            poll_interval=$((poll_interval + 5))
        fi
    done

    local state exit_code
    # sacct may take a moment to finalize; retry briefly
    local retries=5
    while (( retries > 0 )); do
        state="$(sacct -j "$jobid" --format=State --noheader -P | head -1 | tr -d ' ')"
        exit_code="$(sacct -j "$jobid" --format=ExitCode --noheader -P | head -1 | cut -d: -f1)"
        if [[ -n "$state" ]]; then
            break
        fi
        retries=$((retries - 1))
        sleep 2
    done

    log_info "Job ${jobid} finished: state=${state:-UNKNOWN} exit_code=${exit_code:-?}"

    if [[ "${state:-}" == "COMPLETED" && "${exit_code:-1}" == "0" ]]; then
        return 0
    fi
    return "${exit_code:-1}"
}
