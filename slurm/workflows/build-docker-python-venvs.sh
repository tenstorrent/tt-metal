#!/usr/bin/env bash
#SBATCH --job-name=build-python-venvs
#SBATCH --partition=build
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/python-venvs-%j.out
#
# Build Python venv Docker images (ci-build-venv, ci-test-venv) using
# content-addressed tags. Checks if images already exist before building.
#
# Mirrors: .github/workflows/build-docker-python-venvs.yaml
#
# The GHA workflow uses docker buildx bake with dockerfile/docker-bake.hcl.
# In Slurm we do individual docker builds since buildx bake with remote
# builders is not available on bare-metal nodes.
#
# Environment / flags:
#   PIPELINE_ID       (required) Pipeline identifier
#   PLATFORM          Platform string (default: "Ubuntu 22.04")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_config env

require_env PIPELINE_ID

# ---------------------------------------------------------------------------
# Platform parsing
# ---------------------------------------------------------------------------
PLATFORM="${PLATFORM:-Ubuntu 22.04}"

case "${PLATFORM}" in
    "Ubuntu 24.04") UBUNTU_VERSION="24.04"; PYTHON_VERSION="3.12" ;;
    "Ubuntu 22.04"|*) UBUNTU_VERSION="22.04"; PYTHON_VERSION="3.10" ;;
esac

VERSION_NODOT="${UBUNTU_VERSION//.}"
VENV_DOCKERFILE="dockerfile/Dockerfile.python"

log_info "=== Building Python venv images ==="
log_info "  Pipeline:       ${PIPELINE_ID}"
log_info "  Platform:       ${PLATFORM}"
log_info "  Ubuntu version: ${UBUNTU_VERSION}"
log_info "  Python version: ${PYTHON_VERSION}"

require_cmd docker
cd "${REPO_ROOT}"

if [[ ! -f "${VENV_DOCKERFILE}" ]]; then
    log_fatal "Venv Dockerfile not found: ${VENV_DOCKERFILE}"
fi

# ---------------------------------------------------------------------------
# Compute content-addressed tags
# ---------------------------------------------------------------------------
VENV_HASH="$(.github/scripts/dockerfile-hash.sh "${VENV_DOCKERFILE}")"

VENV_BASE="${GHCR_REPO}/python-venv"
CI_BUILD_VENV_TAG="${VENV_BASE}/ci-build:${VERSION_NODOT}-${VENV_HASH}"
CI_TEST_VENV_TAG="${VENV_BASE}/ci-test:${VERSION_NODOT}-${VENV_HASH}"

log_info "  ci-build-venv tag: ${CI_BUILD_VENV_TAG}"
log_info "  ci-test-venv tag:  ${CI_TEST_VENV_TAG}"

docker_login

# ---------------------------------------------------------------------------
# Build & push each venv image if it doesn't already exist
# ---------------------------------------------------------------------------
TAGS_FILE="$(mktemp)"
> "${TAGS_FILE}"

declare -A VENV_TARGETS=(
    ["ci-build-venv"]="${CI_BUILD_VENV_TAG}"
    ["ci-test-venv"]="${CI_TEST_VENV_TAG}"
)

for venv_name in "${!VENV_TARGETS[@]}"; do
    venv_tag="${VENV_TARGETS[$venv_name]}"

    if docker manifest inspect "${venv_tag}" > /dev/null 2>&1; then
        log_info "${venv_name}: already exists, skipping — ${venv_tag}"
    else
        log_info "${venv_name}: building — ${venv_tag}"

        # The target name in the Dockerfile matches the venv name
        docker build \
            --file "${VENV_DOCKERFILE}" \
            --target "${venv_name}" \
            --tag "${venv_tag}" \
            --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
            --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
            --label "org.opencontainers.image.revision=${GIT_SHA}" \
            --pull \
            .

        log_info "Pushing ${venv_tag}"
        docker push "${venv_tag}"
    fi

    # Convert venv name to env var: ci-build-venv -> CI_BUILD_VENV_IMAGE
    ENV_KEY="$(echo "${venv_name}" | tr '-' '_' | tr '[:lower:]' '[:upper:]')_IMAGE"
    echo "${ENV_KEY}=${venv_tag}" >> "${TAGS_FILE}"
done

# ---------------------------------------------------------------------------
# Stage tags to shared storage
# ---------------------------------------------------------------------------
stage_docker_tags "${PIPELINE_ID}" "${TAGS_FILE}"
rm -f "${TAGS_FILE}"

log_info "=== Python venv images complete ==="
