#!/usr/bin/env bash
# push_latest_image.sh - Tag a Docker image and push using regctl.
# Equivalent to .github/actions/push-latest-image-to-ghcr/action.yml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib docker

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") --image IMAGE [OPTIONS]

Tag a Docker image with a new tag (default: latest) and push it to the
registry.  Prefers regctl for server-side copies; falls back to
docker pull/tag/push.

Required:
  --image IMAGE         Full image reference (e.g. ghcr.io/org/repo:sha-abc1234)

Options:
  --registry REGISTRY   Override registry (extracted from --image by default)
  --tag TAG             Target tag (default: latest)
  -h, --help            Show this help message

Environment:
  DOCKER_USERNAME       Registry username (or GITHUB_TOKEN for GHCR)
  DOCKER_PASSWORD       Registry password
  GITHUB_TOKEN          GHCR authentication token
EOF
    exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

IMAGE=""
REGISTRY=""
TARGET_TAG="latest"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image)    IMAGE="$2"; shift 2 ;;
        --registry) REGISTRY="$2"; shift 2 ;;
        --tag)      TARGET_TAG="$2"; shift 2 ;;
        -h|--help)  usage 0 ;;
        *)          log_error "Unknown option: $1"; usage 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ -z "${IMAGE}" ]]; then
    log_error "--image is required"
    usage 1
fi

if [[ ! "${IMAGE}" =~ ^[a-zA-Z0-9._/-]+:[a-zA-Z0-9._-]+$ ]]; then
    log_fatal "Image must be in REPO:TAG format; got '${IMAGE}'"
fi

# ---------------------------------------------------------------------------
# Compute target image
# ---------------------------------------------------------------------------

REPO_NAME="${IMAGE%:*}"

if [[ -n "${REGISTRY}" ]]; then
    # Strip existing registry prefix (everything before first /) and replace
    local_path="${REPO_NAME#*/}"
    LATEST_IMAGE="${REGISTRY}/${local_path}:${TARGET_TAG}"
else
    LATEST_IMAGE="${REPO_NAME}:${TARGET_TAG}"
fi

log_info "Tagging image as ${TARGET_TAG}"
log_info "  Source: ${IMAGE}"
log_info "  Target: ${LATEST_IMAGE}"

# ---------------------------------------------------------------------------
# Login and push
# ---------------------------------------------------------------------------

docker_login

if command -v regctl &>/dev/null; then
    log_info "Using regctl for server-side tag copy"
    regctl image copy "${IMAGE}" "${LATEST_IMAGE}"
else
    log_info "regctl not available; falling back to docker pull/tag/push"
    docker_pull_with_retry "${IMAGE}"
    docker tag "${IMAGE}" "${LATEST_IMAGE}"
    docker push "${LATEST_IMAGE}"
fi

log_info "Image tagged and pushed: ${LATEST_IMAGE}"
