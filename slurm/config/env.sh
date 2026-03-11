#!/usr/bin/env bash
# Common environment variables for all Slurm jobs.
# Source this file from sbatch scripts: source "$(dirname "$0")/../config/env.sh"
set -euo pipefail

# ---------------------------------------------------------------------------
# Container registries
# ---------------------------------------------------------------------------
export GHCR_REGISTRY="ghcr.io"
export GHCR_REPO="${GHCR_REGISTRY}/tenstorrent/tt-metal/tt-metalium"
export HARBOR_REGISTRY="harbor.ci.tenstorrent.net"

# ---------------------------------------------------------------------------
# Site-specific paths (sourced from site.sh)
# ---------------------------------------------------------------------------
# shellcheck source=site.sh
source "${BASH_SOURCE[0]%/*}/site.sh"

# Derived storage paths
export ARTIFACT_BASE="${CI_STORAGE_BASE}/artifacts"
export LOG_BASE="${CI_STORAGE_BASE}/logs"

# ---------------------------------------------------------------------------
# Default Docker image
# ---------------------------------------------------------------------------
# Override DOCKER_IMAGE_TAG in individual job scripts as needed.
export DOCKER_IMAGE_TAG="${DOCKER_IMAGE_TAG:-latest}"
export DOCKER_IMAGE_OS="${DOCKER_IMAGE_OS:-ubuntu-22.04}"
export DOCKER_IMAGE_VARIANT="${DOCKER_IMAGE_VARIANT:-ci-build}"
export DOCKER_IMAGE_ARCH="${DOCKER_IMAGE_ARCH:-amd64}"
# DO NOT set DOCKER_IMAGE here — it must stay empty so that
# resolve_workflow_docker_image() can select the correct variant (dev,
# ci-test, ci-build, …) based on the image_type each workflow requests.
# Setting it here would cause Priority 1 to short-circuit every call.
export DEFAULT_DOCKER_IMAGE="${GHCR_REPO}/${DOCKER_IMAGE_OS}-${DOCKER_IMAGE_VARIANT}-${DOCKER_IMAGE_ARCH}:${DOCKER_IMAGE_TAG}"

# ---------------------------------------------------------------------------
# In-container paths and build variables
# ---------------------------------------------------------------------------
export TT_METAL_HOME="${TT_METAL_HOME:-${CONTAINER_WORKDIR}}"
export PYTHONPATH="${PYTHONPATH:-${TT_METAL_HOME}}"
export ARCH_NAME="${ARCH_NAME:-wormhole_b0}"
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"

# CI flags
export TT_GH_CI_INFRA="${TT_GH_CI_INFRA:-1}"

# Model/cache paths (mounted read-only from Weka/MLPerf)
export HF_HUB_CACHE="${HF_HUB_CACHE:-${MLPERF_BASE}/huggingface/hub}"
export HF_HOME="${HF_HOME:-${MLPERF_BASE}/huggingface}"
export TT_CACHE_HOME="${TT_CACHE_HOME:-${MLPERF_BASE}/huggingface/tt_cache}"

# ---------------------------------------------------------------------------
# Container device mounts & volumes
# ---------------------------------------------------------------------------
# Devices that must be passed through to every hardware job container.
CONTAINER_DEVICES=(
    "${TT_DEVICE_PATH}"
)

# Volume mounts shared by all jobs.
CONTAINER_VOLUMES=(
    "${HUGEPAGES_PATH}:${HUGEPAGES_PATH}"
    "/etc/passwd:/etc/passwd:ro"
    "/etc/shadow:/etc/shadow:ro"
    "/etc/bashrc:/etc/bashrc:ro"
    "${MLPERF_BASE}:${MLPERF_BASE}:ro"
)

# Build a docker-run device string from the arrays above.
# Usage: docker run $(build_docker_device_args) ...
build_docker_device_args() {
    local args=""
    for dev in "${CONTAINER_DEVICES[@]}"; do
        args+=" --device ${dev}"
    done
    for vol in "${CONTAINER_VOLUMES[@]}"; do
        args+=" -v ${vol}"
    done
    echo "${args}"
}

# Full docker-run invocation fragment (devices + common options).
# Usage: docker run $(build_docker_run_opts [workdir]) <image> <cmd>
build_docker_run_opts() {
    local workdir="${1:-${TT_METAL_HOME}}"
    local uid; uid="$(id -u)"
    local gid; gid="$(id -g)"

    local opts=""
    opts+=" --rm"
    opts+=" --net=host"
    opts+=" --log-driver local --log-opt max-size=50m"
    opts+=" -u ${uid}:${gid}"
    opts+=" -w ${workdir}"
    opts+=" -e TT_METAL_HOME=${TT_METAL_HOME}"
    opts+=" -e PYTHONPATH=${PYTHONPATH}"
    opts+=" -e ARCH_NAME=${ARCH_NAME}"
    opts+=" -e LOGURU_LEVEL=${LOGURU_LEVEL}"
    opts+=" -e TT_GH_CI_INFRA=${TT_GH_CI_INFRA}"
    opts+=" -e HOME=${workdir}"
    opts+="$(build_docker_device_args)"
    echo "${opts}"
}

# Variant: skip hugepages mount for viommu runners.
build_docker_run_opts_viommu() {
    local workdir="${1:-${TT_METAL_HOME}}"
    # Temporarily remove hugepages from CONTAINER_VOLUMES
    local saved=("${CONTAINER_VOLUMES[@]}")
    local filtered=()
    for vol in "${CONTAINER_VOLUMES[@]}"; do
        [[ "${vol}" == */hugepages* ]] && continue
        filtered+=("${vol}")
    done
    CONTAINER_VOLUMES=("${filtered[@]}")
    build_docker_run_opts "${workdir}"
    CONTAINER_VOLUMES=("${saved[@]}")
}
