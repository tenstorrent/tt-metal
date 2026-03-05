#!/usr/bin/env bash
#SBATCH --job-name=build-docker-artifact
#SBATCH --partition=build
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/docker-build-%A_%a.out
#SBATCH --array=0-5
#
# Build Docker images as a job array. Each SLURM_ARRAY_TASK_ID builds one image:
#   0=ci-build  1=ci-test  2=dev  3=basic-dev  4=basic-ttnn  5=manylinux
#
# Mirrors: .github/workflows/build-docker-artifact.yaml
#
# Each array task:
#   1. Computes a content-addressed tag via dockerfile-hash.sh
#   2. Checks if the image already exists in the registry (skips if so)
#   3. Builds and pushes the image
#   4. Stages the tag to shared storage for downstream jobs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_config env

TASK_ID="$(get_array_task_id)"

# ---------------------------------------------------------------------------
# Image definitions — parallels the GHA matrix
# ---------------------------------------------------------------------------
declare -a IMAGE_TYPES=(ci-build ci-test dev basic-dev basic-ttnn manylinux)
declare -a DOCKER_TARGETS=(ci-build ci-test dev base basic-ttnn-runtime "")
declare -a DOCKERFILES=(
    dockerfile/Dockerfile
    dockerfile/Dockerfile
    dockerfile/Dockerfile
    dockerfile/Dockerfile.basic-dev
    dockerfile/Dockerfile.basic-dev
    dockerfile/Dockerfile.manylinux
)

IMAGE_TYPE="${IMAGE_TYPES[$TASK_ID]}"
DOCKER_TARGET="${DOCKER_TARGETS[$TASK_ID]}"
DOCKERFILE="${DOCKERFILES[$TASK_ID]}"

PLATFORM="${PLATFORM:-Ubuntu 22.04}"
ARCH="${DOCKER_IMAGE_ARCH:-amd64}"

# ---------------------------------------------------------------------------
# Parse platform -> distro + version + python version
# ---------------------------------------------------------------------------
case "${PLATFORM}" in
    "Ubuntu 24.04") DISTRO="ubuntu"; VERSION="24.04"; PYTHON_VERSION="3.12" ;;
    "Ubuntu 22.04") DISTRO="ubuntu"; VERSION="22.04"; PYTHON_VERSION="3.10" ;;
    *)              DISTRO="ubuntu"; VERSION="22.04"; PYTHON_VERSION="3.10" ;;
esac

# ---------------------------------------------------------------------------
# Compute content-addressed tag
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"

EXTRA_FILES=".github/workflows/build-docker-artifact.yaml"

case "${IMAGE_TYPE}" in
    ci-build|ci-test|dev)
        HASH="$(.github/scripts/dockerfile-hash.sh "${DOCKERFILE}" "${EXTRA_FILES}")"
        IMAGE_NAME="${DISTRO}-${VERSION}-${IMAGE_TYPE}-${ARCH}"
        ;;
    basic-dev|basic-ttnn)
        HASH="$(.github/scripts/dockerfile-hash.sh "${DOCKERFILE}")"
        IMAGE_NAME="${DISTRO}-${VERSION}-${IMAGE_TYPE}-${ARCH}"
        ;;
    manylinux)
        HASH="$(.github/scripts/dockerfile-hash.sh "${DOCKERFILE}" "${EXTRA_FILES}")"
        IMAGE_NAME="manylinux-${ARCH}"
        ;;
esac

FULL_TAG="${GHCR_REPO}/${IMAGE_NAME}:${HASH}"

log_info "Task ${TASK_ID}/${#IMAGE_TYPES[@]}: ${IMAGE_TYPE}"
log_info "  Dockerfile: ${DOCKERFILE}"
log_info "  Target:     ${DOCKER_TARGET:-<none>}"
log_info "  Tag:        ${FULL_TAG}"

# ---------------------------------------------------------------------------
# Check if image already exists (skip expensive build)
# ---------------------------------------------------------------------------
require_cmd docker
docker_login

if docker manifest inspect "${FULL_TAG}" > /dev/null 2>&1; then
    log_info "Image already exists, skipping build: ${FULL_TAG}"
else
    log_info "Image not found — building"

    declare -a BUILD_ARGS=(
        --file "${DOCKERFILE}"
        --tag "${FULL_TAG}"
        --label "org.opencontainers.image.revision=${GIT_SHA}"
        --label "org.opencontainers.image.source=https://github.com/tenstorrent/tt-metal"
        --pull
    )

    if [[ -n "${DOCKER_TARGET}" ]]; then
        BUILD_ARGS+=(--target "${DOCKER_TARGET}")
    fi

    # Build args vary by image type
    case "${IMAGE_TYPE}" in
        ci-build|ci-test|dev|basic-dev|basic-ttnn)
            BUILD_ARGS+=(
                --build-arg "UBUNTU_VERSION=${VERSION}"
                --build-arg "PYTHON_VERSION=${PYTHON_VERSION}"
            )
            ;;
    esac

    docker build "${BUILD_ARGS[@]}" .

    log_info "Pushing ${FULL_TAG}"
    docker push "${FULL_TAG}"
fi

# ---------------------------------------------------------------------------
# Stage tag to shared storage for downstream jobs
# ---------------------------------------------------------------------------
TAGS_FILE="$(mktemp)"
ENV_VAR_NAME="$(echo "${IMAGE_TYPE}" | tr '-' '_' | tr '[:lower:]' '[:upper:]')_IMAGE"
echo "${ENV_VAR_NAME}=${FULL_TAG}" > "${TAGS_FILE}"

stage_docker_tags "${PIPELINE_ID}" "${TAGS_FILE}"
rm -f "${TAGS_FILE}"

log_info "Docker image ${IMAGE_TYPE} complete: ${FULL_TAG}"
