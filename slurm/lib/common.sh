#!/usr/bin/env bash
# common.sh - Core shared library sourced by all Slurm CI scripts.
# Provides logging, environment validation, git context, pipeline ID
# generation, and sourcing helpers.

set -euo pipefail

# Guard against double-sourcing
[[ -n "${_SLURM_CI_COMMON_SH:-}" ]] && return 0
_SLURM_CI_COMMON_SH=1

# ---------------------------------------------------------------------------
# Script directory detection
# ---------------------------------------------------------------------------

SLURM_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly SLURM_SCRIPTS_DIR
export SLURM_SCRIPTS_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log() {
    local level="$1"; shift
    local ts
    ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf '[%s] %s %s\n' "$level" "$ts" "$*" >&2
}

log_info()  { _log INFO  "$@"; }
log_warn()  { _log WARN  "$@"; }
log_error() { _log ERROR "$@"; }

log_fatal() {
    _log FATAL "$@"
    exit 1
}

# ---------------------------------------------------------------------------
# Git context
# ---------------------------------------------------------------------------

GIT_SHA="${GIT_SHA:-$(git rev-parse HEAD 2>/dev/null || echo 'unknown')}"
GIT_REF="${GIT_REF:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')}"
GIT_SHORT_SHA="${GIT_SHA:0:7}"
export GIT_SHA GIT_REF GIT_SHORT_SHA

# ---------------------------------------------------------------------------
# Pipeline ID
# ---------------------------------------------------------------------------

generate_pipeline_id() {
    local ts short
    ts="$(date -u '+%Y%m%d-%H%M%S')"
    short="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    printf '%s-%s' "$ts" "$short"
}

PIPELINE_ID="${PIPELINE_ID:-$(generate_pipeline_id)}"
export PIPELINE_ID

# ---------------------------------------------------------------------------
# Workspace and repo root
# ---------------------------------------------------------------------------

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
readonly REPO_ROOT
export REPO_ROOT

WORKSPACE="${WORKSPACE:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
export WORKSPACE

# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

require_env() {
    local var="$1"
    if [[ -z "${!var:-}" ]]; then
        log_fatal "Required environment variable '$var' is not set or empty"
    fi
}

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" &>/dev/null; then
        log_fatal "Required command '$cmd' not found in PATH"
    fi
}

# ---------------------------------------------------------------------------
# Cleanup trap
# ---------------------------------------------------------------------------

_cleanup_handlers=()

register_cleanup() {
    _cleanup_handlers+=("$1")
}

cleanup_on_exit() {
    local rc=$?
    for handler in "${_cleanup_handlers[@]+"${_cleanup_handlers[@]}"}"; do
        log_info "Running cleanup: $handler"
        eval "$handler" || log_warn "Cleanup handler failed: $handler"
    done
    return "$rc"
}

trap cleanup_on_exit EXIT

# ---------------------------------------------------------------------------
# Slurm introspection
# ---------------------------------------------------------------------------

is_slurm_job() {
    [[ -n "${SLURM_JOB_ID:-}" ]]
}

get_job_name() {
    printf '%s' "${SLURM_JOB_NAME:-local}"
}

get_array_task_id() {
    printf '%s' "${SLURM_ARRAY_TASK_ID:-0}"
}

# Convenience aliases so library code can use a uniform SLURM_CI_ prefix
# without worrying about which native SLURM_ variable names exist.
export SLURM_CI_JOB_ID="${SLURM_JOB_ID:-0}"
export SLURM_CI_JOB_NAME="${SLURM_JOB_NAME:-local}"
export SLURM_CI_NODELIST="${SLURM_NODELIST:-${SLURM_JOB_NODELIST:-localhost}}"
export SLURM_CI_ARRAY_TASK="${SLURM_ARRAY_TASK_ID:-0}"

# ---------------------------------------------------------------------------
# parse_common_args - Parse CLI flags shared across workflow scripts
# ---------------------------------------------------------------------------
# When workflows are submitted via sbatch, $@ is empty and this is a no-op.
# When run directly for testing, it accepts the same flags as submit.sh.

parse_common_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --docker-image)  export DOCKER_IMAGE="${2:?--docker-image requires a value}"; shift 2 ;;
            --arch)          export ARCH_NAME="${2:?--arch requires a value}"; shift 2 ;;
            --ref)           export GIT_REF="${2:?--ref requires a value}"; shift 2 ;;
            --pipeline-id)   export PIPELINE_ID="${2:?--pipeline-id requires a value}"; shift 2 ;;
            --partition)     shift 2 ;;  # consumed by submit.sh, ignored here
            --)              break ;;
            *)               log_warn "Unknown arg passed to workflow: $1"; shift ;;
        esac
    done
}

# ---------------------------------------------------------------------------
# Sourcing helpers
# ---------------------------------------------------------------------------

source_lib() {
    local lib_name="$1"
    local lib_path="${SLURM_SCRIPTS_DIR}/lib/${lib_name}.sh"
    if [[ ! -f "$lib_path" ]]; then
        log_fatal "Library not found: $lib_path"
    fi
    # shellcheck source=/dev/null
    source "$lib_path"
}

source_config() {
    local config_name="$1"
    local config_path="${SLURM_SCRIPTS_DIR}/config/${config_name}.sh"
    if [[ ! -f "$config_path" ]]; then
        log_fatal "Config not found: $config_path"
    fi
    # shellcheck source=/dev/null
    source "$config_path"
}
