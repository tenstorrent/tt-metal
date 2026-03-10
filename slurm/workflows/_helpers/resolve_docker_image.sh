#!/usr/bin/env bash
# resolve_docker_image.sh - Resolve Docker image tag for workflow jobs
#
# Source this file, then call resolve_workflow_docker_image.
# Sets: DOCKER_IMAGE (exported)

# Guard against double-sourcing
[[ -n "${_SLURM_HELPERS_RESOLVE_DOCKER_SH:-}" ]] && return 0
_SLURM_HELPERS_RESOLVE_DOCKER_SH=1

SLURM_CI_LIB_DIR="${SLURM_CI_LIB_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../lib" && pwd)}"
# shellcheck source=../../lib/common.sh
source "${SLURM_CI_LIB_DIR}/common.sh"
source_config env

# ---------------------------------------------------------------------------
# resolve_workflow_docker_image [image_type]
# ---------------------------------------------------------------------------
# Resolves the Docker image for a workflow job.
#
# Resolution order:
#   1. DOCKER_IMAGE env var (set by --docker-image flag or caller)
#   2. Staged docker tags from the pipeline's artifact directory
#   3. Default constructed from DOCKER_IMAGE_TAG + env.sh defaults
#
# After resolution, prefers the Harbor on-prem mirror when reachable.
#
# Args:
#   image_type  One of: ci-build ci-test dev basic-dev basic-ttnn manylinux
#               Defaults to "dev".
# Exports:
#   DOCKER_IMAGE
resolve_workflow_docker_image() {
    local image_type="${1:-dev}"

    # --- Priority 1: explicit env var (e.g. from --docker-image CLI flag) ---
    if [[ -n "${DOCKER_IMAGE:-}" ]]; then
        log_info "Docker image from environment: ${DOCKER_IMAGE}"
        _try_harbor_mirror
        export DOCKER_IMAGE
        return 0
    fi

    # --- Priority 2: staged tags from a prior docker-build job ---
    local tags_file="${ARTIFACT_DIR:-${ARTIFACT_BASE}/${PIPELINE_ID}}/docker/image_tags.env"
    if [[ -f "${tags_file}" ]]; then
        log_info "Loading staged docker tags from ${tags_file}"
        # shellcheck disable=SC1090
        source "${tags_file}"

        case "${image_type}" in
            ci-build)   DOCKER_IMAGE="${CI_BUILD_IMAGE:-}" ;;
            ci-test)    DOCKER_IMAGE="${CI_TEST_IMAGE:-}" ;;
            dev)        DOCKER_IMAGE="${DEV_IMAGE:-}" ;;
            basic-dev)  DOCKER_IMAGE="${BASIC_DEV_IMAGE:-}" ;;
            basic-ttnn) DOCKER_IMAGE="${BASIC_TTNN_IMAGE:-}" ;;
            manylinux)  DOCKER_IMAGE="${MANYLINUX_IMAGE:-}" ;;
            *)          DOCKER_IMAGE="${DEV_IMAGE:-}" ;;
        esac

        if [[ -n "${DOCKER_IMAGE:-}" ]]; then
            log_info "Docker image from staged tags (${image_type}): ${DOCKER_IMAGE}"
            _try_harbor_mirror
            export DOCKER_IMAGE
            return 0
        fi
        log_warn "Staged tags found but no image for type '${image_type}'"
    fi

    # --- Priority 3: construct from env.sh defaults ---
    local tag="${DOCKER_IMAGE_TAG:-latest}"
    local os="${DOCKER_IMAGE_OS:-ubuntu-22.04}"
    local arch="${DOCKER_IMAGE_ARCH:-amd64}"

    local variant
    case "${image_type}" in
        ci-build)   variant="ci-build" ;;
        ci-test)    variant="ci-test" ;;
        dev)        variant="dev" ;;
        basic-dev)  variant="basic-dev" ;;
        basic-ttnn) variant="basic-ttnn" ;;
        manylinux)  variant="manylinux"; os="manylinux" ;;
        *)          variant="${DOCKER_IMAGE_VARIANT:-dev}" ;;
    esac

    DOCKER_IMAGE="${GHCR_REPO}/${os}-${variant}-${arch}:${tag}"
    log_info "Docker image from defaults (${image_type}): ${DOCKER_IMAGE}"
    _try_harbor_mirror
    export DOCKER_IMAGE
}

# ---------------------------------------------------------------------------
# _try_harbor_mirror - Swap GHCR image for Harbor when on-prem and reachable
# ---------------------------------------------------------------------------
# Harbor pulls are significantly faster from CI nodes on the internal network.
# Silently falls back to the GHCR image when Harbor is unreachable.

_try_harbor_mirror() {
    # Only relevant for GHCR images
    [[ "${DOCKER_IMAGE}" == "${GHCR_REGISTRY}"* ]] || return 0
    # Opt-out via PREFER_HARBOR=0
    [[ "${PREFER_HARBOR:-1}" != "0" ]] || return 0

    if timeout 3 bash -c "echo >/dev/tcp/${HARBOR_REGISTRY}/443" 2>/dev/null; then
        local harbor_image="${DOCKER_IMAGE/${GHCR_REGISTRY}/${HARBOR_REGISTRY}}"
        log_info "Harbor reachable — using mirror: ${harbor_image}"
        DOCKER_IMAGE="${harbor_image}"
    fi
}

# Short alias used by some workflow scripts
resolve_docker_image() { resolve_workflow_docker_image "$@"; }
