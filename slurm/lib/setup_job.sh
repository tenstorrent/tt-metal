#!/usr/bin/env bash
# setup_job.sh - Job prologue: workspace preparation and environment setup
# Equivalent to .github/actions/setup-job/action.yml

set -euo pipefail

# Guard against double-sourcing
[[ -n "${_SLURM_CI_SETUP_JOB_SH:-}" ]] && return 0
_SLURM_CI_SETUP_JOB_SH=1

SLURM_CI_LIB_DIR="${SLURM_CI_LIB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SLURM_CI_LIB_DIR}/common.sh"
source "${SLURM_CI_LIB_DIR}/artifacts.sh"

# ---------------------------------------------------------------------------
# _reset_tt_devices - Reset Tenstorrent accelerators on all job nodes
# ---------------------------------------------------------------------------
# Mirrors the GitHub Actions runner pre-job reset (tt-smi -r).
# Skipped automatically when tt-smi is not available or no TT device exists.
# Set SKIP_DEVICE_RESET=1 to disable (e.g., for build-only jobs).
_reset_tt_devices() {
    if [[ "${SKIP_DEVICE_RESET:-0}" == "1" ]]; then
        return 0
    fi

    if ! command -v tt-smi &>/dev/null; then
        log_debug "tt-smi not found, skipping device reset"
        return 0
    fi

    if [[ ! -e "${TT_DEVICE_PATH:-/dev/tenstorrent}" ]]; then
        log_debug "No Tenstorrent device found, skipping device reset"
        return 0
    fi

    local num_nodes="${SLURM_JOB_NUM_NODES:-1}"

    if (( num_nodes > 1 )) && command -v srun &>/dev/null; then
        log_info "Resetting TT devices on ${num_nodes} nodes (srun)..."
        srun --ntasks-per-node=1 bash -c \
            'echo "[$(hostname)] tt-smi -r" && tt-smi -r' 2>&1 | \
            while IFS= read -r line; do log_info "  ${line}"; done || \
            log_warn "tt-smi -r returned non-zero on one or more nodes (rc=$?), continuing"
    else
        log_info "Resetting TT devices on $(hostname)..."
        tt-smi -r 2>&1 | \
            while IFS= read -r line; do log_info "  ${line}"; done || \
            log_warn "tt-smi -r failed (rc=$?), continuing"
    fi
}

# ---------------------------------------------------------------------------
# setup_job - Main prologue function
# ---------------------------------------------------------------------------
# Controlled by environment variables:
#   PIPELINE_ID               (required) Pipeline identifier
#   JOB_WORKSPACE             Host-side working directory for build artifacts.
#                             Defaults to ${ARTIFACT_DIR}/workspace (on NFS).
#                             NOTE: this is NOT the Docker container workdir;
#                             CONTAINER_WORKDIR is a separate Docker-only setting.
#   BUILD_ARTIFACT            Set to 1 to fetch & extract build tarball
#   INSTALL_WHEEL             Set to 1 to fetch & install Python wheel
#   ENABLE_WATCHER            Set to 1 to enable TT Metal watcher
#   ENABLE_LIGHTWEIGHT_ASSERTS Set to 1 to enable lightweight kernel asserts
#   ENABLE_LLK_ASSERTS        Set to 1 to enable LLK asserts
#   ENABLE_KERNEL_CCACHE      Set to 1 to enable kernel ccache
#   CCACHE_REMOTE_STORAGE     Redis URL for kernel ccache (required if ccache enabled)
setup_job() {
    require_env PIPELINE_ID

    local workspace="${JOB_WORKSPACE:-${ARTIFACT_DIR:-/tmp/slurm-ci-${PIPELINE_ID}}/workspace}"

    log_info "=== Job setup starting ==="
    log_info "Pipeline:  ${PIPELINE_ID}"
    log_info "Workspace: ${workspace}"
    log_info "Node:      ${SLURM_CI_NODELIST}"
    log_info "Job:       ${SLURM_CI_JOB_NAME} (${SLURM_CI_JOB_ID}/${SLURM_CI_ARRAY_TASK})"

    # -- Reset TT devices (mirrors GH Actions runner pre-job reset) --
    _reset_tt_devices

    # -- Create workspace --
    mkdir -p "$workspace"

    # -- Fetch build artifact --
    if [[ "${BUILD_ARTIFACT:-0}" == "1" ]]; then
        local tarball
        tarball="$(get_artifact_dir "$PIPELINE_ID")/build/ttm_any.tar.zst"
        if [[ -f "$tarball" ]]; then
            log_info "Fetching build artifact..."
            require_cmd tar
            fetch_build_artifact "$PIPELINE_ID" "$workspace"
        else
            log_info "No build tarball found; running from existing workspace (${workspace})"
        fi
    fi

    # -- Install wheel --
    if [[ "${INSTALL_WHEEL:-0}" == "1" ]]; then
        log_info "Installing Python wheel..."
        require_cmd uv
        fetch_wheel "$PIPELINE_ID"
    fi

    # -- Watcher --
    if [[ "${ENABLE_WATCHER:-0}" == "1" ]]; then
        log_info "Enabling TT Metal Watcher"
        export TT_METAL_WATCHER=1
        export TT_METAL_WATCHER_APPEND=1
        export TT_METAL_WATCHER_NOINLINE=1
    fi

    # -- Lightweight kernel asserts --
    if [[ "${ENABLE_LIGHTWEIGHT_ASSERTS:-0}" == "1" ]]; then
        log_info "Enabling lightweight kernel asserts"
        export TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1
    fi

    # -- LLK asserts --
    if [[ "${ENABLE_LLK_ASSERTS:-0}" == "1" ]]; then
        log_info "Enabling LLK asserts"
        export TT_METAL_LLK_ASSERTS=1
    fi

    # -- Kernel ccache --
    if [[ "${ENABLE_KERNEL_CCACHE:-0}" == "1" ]]; then
        log_info "Enabling kernel ccache"
        if [[ -z "${CCACHE_REMOTE_STORAGE:-}" ]]; then
            log_error "CCACHE_REMOTE_STORAGE must be set when kernel ccache is enabled"
            return 1
        fi
        export CCACHE_COMPILERCHECK=content
        export CCACHE_REMOTE_ONLY=true
        export CCACHE_NOHASHDIR=true
        export TT_METAL_CCACHE_KERNEL_SUPPORT=1
        export CCACHE_TEMPDIR=/tmp/ccache
        mkdir -p /tmp/ccache
    fi

    # -- Summary --
    log_info "=== Job setup complete ==="
    log_info "  BUILD_ARTIFACT=${BUILD_ARTIFACT:-0}"
    log_info "  INSTALL_WHEEL=${INSTALL_WHEEL:-0}"
    log_info "  ENABLE_WATCHER=${ENABLE_WATCHER:-0}"
    log_info "  ENABLE_LIGHTWEIGHT_ASSERTS=${ENABLE_LIGHTWEIGHT_ASSERTS:-0}"
    log_info "  ENABLE_LLK_ASSERTS=${ENABLE_LLK_ASSERTS:-0}"
    log_info "  ENABLE_KERNEL_CCACHE=${ENABLE_KERNEL_CCACHE:-0}"
}
