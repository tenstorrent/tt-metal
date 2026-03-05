#!/usr/bin/env bash
#SBATCH --job-name=publish-release-image
#SBATCH --partition=build
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err
#
# Push release Docker image to GHCR.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib docker
source_config env

require_env PIPELINE_ID
require_env RELEASE_VERSION

RELEASE_TAG="${RELEASE_VERSION}"
SOURCE_IMAGE="${DOCKER_IMAGE:-${DEFAULT_DOCKER_IMAGE}}"
RELEASE_IMAGE="${GHCR_REPO}/${DOCKER_IMAGE_OS}-release-${DOCKER_IMAGE_ARCH}:${RELEASE_TAG}"

log_info "=== Publish release image starting ==="
log_info "  Source:  ${SOURCE_IMAGE}"
log_info "  Target:  ${RELEASE_IMAGE}"
log_info "  Version: ${RELEASE_VERSION}"

require_cmd docker
docker_login

docker_pull_with_retry "${SOURCE_IMAGE}"

docker tag "${SOURCE_IMAGE}" "${RELEASE_IMAGE}"

LATEST_IMAGE="${GHCR_REPO}/${DOCKER_IMAGE_OS}-release-${DOCKER_IMAGE_ARCH}:latest"
docker tag "${SOURCE_IMAGE}" "${LATEST_IMAGE}"

log_info "Pushing ${RELEASE_IMAGE}"
docker push "${RELEASE_IMAGE}"

log_info "Pushing ${LATEST_IMAGE}"
docker push "${LATEST_IMAGE}"

TAGS_FILE="$(mktemp)"
cat > "${TAGS_FILE}" <<EOF
RELEASE_IMAGE=${RELEASE_IMAGE}
RELEASE_IMAGE_LATEST=${LATEST_IMAGE}
EOF

stage_docker_tags "${PIPELINE_ID}" "${TAGS_FILE}"
rm -f "${TAGS_FILE}"

log_info "=== Publish release image complete ==="
