#!/usr/bin/env bash
# submit_dependent.sh - Dependency-aware job submission helpers
#
# Source this file, then call the helper functions.
# All functions print the new JOBID to stdout and log the submission.

set -euo pipefail

# Guard against double-sourcing
[[ -n "${_SLURM_HELPERS_SUBMIT_DEP_SH:-}" ]] && return 0
_SLURM_HELPERS_SUBMIT_DEP_SH=1

SLURM_CI_LIB_DIR="${SLURM_CI_LIB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../lib" && pwd)}"
# shellcheck source=../../lib/common.sh
source "${SLURM_CI_LIB_DIR}/common.sh"

require_cmd sbatch

# ---------------------------------------------------------------------------
# _submit_with_dep <dep_type> <dep_id> <script> [extra_sbatch_args...]
# ---------------------------------------------------------------------------
# Internal: submit a single job with an optional Slurm dependency.
# dep_type: afterok, afterany, etc.
# dep_id:   parent job ID (empty or "0" means no dependency)

_submit_with_dep() {
    local dep_type="$1"
    local dep_id="$2"
    local script="$3"
    shift 3

    local -a sbatch_args=(
        --parsable
        "--export=ALL,PIPELINE_ID=${PIPELINE_ID}"
    )

    if [[ -n "${dep_id}" && "${dep_id}" != "0" ]]; then
        sbatch_args+=("--dependency=${dep_type}:${dep_id}")
    fi

    if [[ -n "${LOG_DIR:-}" ]]; then
        sbatch_args+=(
            "--output=${LOG_DIR}/%x-%j.out"
            "--error=${LOG_DIR}/%x-%j.err"
        )
    fi

    if [[ $# -gt 0 ]]; then
        sbatch_args+=("$@")
    fi
    sbatch_args+=("${script}")

    local output
    output="$(sbatch "${sbatch_args[@]}")"
    local jobid="${output%%_*}"

    if [[ -z "${jobid}" ]]; then
        log_error "Failed to submit ${script} (${dep_type}:${dep_id})"
        return 1
    fi

    log_info "Submitted ${script} as job ${jobid} (${dep_type}:${dep_id:-none})"
    echo "${jobid}"
}

# ---------------------------------------------------------------------------
# submit_after <parent_jobid> <script> [extra_sbatch_args...]
# ---------------------------------------------------------------------------
# Submit a job with --dependency=afterok:$parent_jobid.
# The job runs only if the parent exits successfully.

submit_after() {
    local parent_jobid="${1:?parent_jobid required}"
    local script="${2:?script required}"
    shift 2
    _submit_with_dep "afterok" "$parent_jobid" "$script" "$@"
}

# ---------------------------------------------------------------------------
# submit_after_any <parent_jobid> <script> [extra_sbatch_args...]
# ---------------------------------------------------------------------------
# Submit a job with --dependency=afterany:$parent_jobid.
# The job runs regardless of the parent's exit status.

submit_after_any() {
    local parent_jobid="${1:?parent_jobid required}"
    local script="${2:?script required}"
    shift 2
    _submit_with_dep "afterany" "$parent_jobid" "$script" "$@"
}

# ---------------------------------------------------------------------------
# submit_chain <script1> <script2> [script3 ...]
# ---------------------------------------------------------------------------
# Submit a linear chain: each script depends on the previous (afterok).
# Prints every JOBID (one per line); the final line is the tail of the chain.

submit_chain() {
    if [[ $# -lt 2 ]]; then
        log_error "submit_chain requires at least 2 scripts"
        return 1
    fi

    local prev_jobid=""
    local jobid=""

    for script in "$@"; do
        if [[ -z "$prev_jobid" ]]; then
            jobid="$(_submit_with_dep "afterok" "" "$script")"
        else
            jobid="$(_submit_with_dep "afterok" "$prev_jobid" "$script")"
        fi
        prev_jobid="$jobid"
    done

    # Return the last job ID (callers needing all IDs can capture full output)
    echo "$jobid"
}

# ---------------------------------------------------------------------------
# fan_out <parent_jobid> <script1> <script2> [script3 ...]
# ---------------------------------------------------------------------------
# Submit multiple independent jobs, each depending on parent_jobid (afterok).
# Prints each JOBID on its own line.

fan_out() {
    local parent_jobid="${1:?parent_jobid required}"
    shift

    if [[ $# -lt 1 ]]; then
        log_error "fan_out requires at least one script after parent_jobid"
        return 1
    fi

    for script in "$@"; do
        _submit_with_dep "afterok" "$parent_jobid" "$script"
    done
}

# ---------------------------------------------------------------------------
# collect <parent_jobids> <script> [extra_sbatch_args...]
# ---------------------------------------------------------------------------
# Submit one job that depends on ALL parent jobs completing successfully.
#
# parent_jobids: either a colon-separated string ("123:456:789") or the name
#                of a bash array variable containing the IDs.
#
# Examples:
#   collect "100:200:300" ./report.sh
#
#   local -a parents=(100 200 300)
#   collect parents ./report.sh

collect() {
    local deps_input="${1:?parent job IDs required}"
    local script="${2:?script required}"
    shift 2

    local dep_string=""

    # Detect whether deps_input is the name of an array variable
    if declare -p "$deps_input" &>/dev/null \
       && [[ "$(declare -p "$deps_input" 2>/dev/null)" == "declare -a"* ]]; then
        local -n _deps_arr="$deps_input"
        dep_string="$(IFS=:; echo "${_deps_arr[*]}")"
    else
        dep_string="$deps_input"
    fi

    if [[ -z "$dep_string" ]]; then
        log_error "collect: no parent job IDs provided"
        return 1
    fi

    local -a sbatch_args=(
        --parsable
        "--dependency=afterok:${dep_string}"
        "--export=ALL,PIPELINE_ID=${PIPELINE_ID}"
    )

    if [[ -n "${LOG_DIR:-}" ]]; then
        sbatch_args+=(
            "--output=${LOG_DIR}/%x-%j.out"
            "--error=${LOG_DIR}/%x-%j.err"
        )
    fi

    if [[ $# -gt 0 ]]; then
        sbatch_args+=("$@")
    fi
    sbatch_args+=("${script}")

    local output
    output="$(sbatch "${sbatch_args[@]}")"
    local jobid="${output%%_*}"

    if [[ -z "$jobid" ]]; then
        log_error "Failed to submit collector ${script} (afterok:${dep_string})"
        return 1
    fi

    log_info "Submitted collector ${script} as job ${jobid} (afterok:${dep_string})"
    echo "${jobid}"
}
