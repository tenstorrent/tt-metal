#!/usr/bin/env bash
#SBATCH --job-name=build-eval-image
#SBATCH --partition=build
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/eval-image-%j.out
#
# Build the evaluation Docker image for model benchmarking and scoring.
#
# Mirrors: .github/workflows/build-evaluation-image.yaml
#
# The evaluation image is built from dockerfile/Dockerfile.evaluation and
# pushed to Harbor (primary) and/or GHCR. Supports content-addressed tagging
# and BuildKit remote builders when available.
#
# Environment / flags:
#   PIPELINE_ID       (required) Pipeline identifier
#   EVAL_IMAGE_TAG    Override image tag (default: <sha>-<pipeline_id>)
#   BUILD_ARGS        Extra docker build-args (space-separated KEY=VALUE)
#   PUSH_TO_HARBOR    true/false — push to Harbor registry (default: true)
#   PUSH_TO_GHCR      true/false — push to GHCR (default: false)
#   CACHE_FROM         Docker cache-from spec
#   CACHE_TO           Docker cache-to spec

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_config env

require_env PIPELINE_ID

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EVAL_DOCKERFILE="dockerfile/Dockerfile.evaluation"
PUSH_TO_HARBOR="${PUSH_TO_HARBOR:-true}"
PUSH_TO_GHCR="${PUSH_TO_GHCR:-false}"
CACHE_FROM="${CACHE_FROM:-}"
CACHE_TO="${CACHE_TO:-}"

EVAL_IMAGE_TAG="${EVAL_IMAGE_TAG:-${GIT_SHA}-${PIPELINE_ID}}"

log_info "=== Building evaluation image ==="
log_info "  Pipeline:   ${PIPELINE_ID}"
log_info "  Dockerfile: ${EVAL_DOCKERFILE}"
log_info "  Tag:        ${EVAL_IMAGE_TAG}"
log_info "  Harbor:     ${PUSH_TO_HARBOR}"
log_info "  GHCR:       ${PUSH_TO_GHCR}"

require_cmd docker
cd "${REPO_ROOT}"

if [[ ! -f "${EVAL_DOCKERFILE}" ]]; then
    log_fatal "Evaluation Dockerfile not found: ${EVAL_DOCKERFILE}"
fi

# ---------------------------------------------------------------------------
# Compute image tags for each target registry
# ---------------------------------------------------------------------------
declare -a IMAGE_TAGS=()
declare -a PUSH_TAGS=()

if [[ "${PUSH_TO_HARBOR}" == "true" ]]; then
    HARBOR_TAG="${HARBOR_REGISTRY}/evaluation/tt-metalium:${EVAL_IMAGE_TAG}"
    IMAGE_TAGS+=("${HARBOR_TAG}")
    PUSH_TAGS+=("${HARBOR_TAG}")

    docker_login "${HARBOR_REGISTRY}"
fi

if [[ "${PUSH_TO_GHCR}" == "true" ]]; then
    GHCR_TAG="ghcr.io/tenstorrent/evaluation/tt-metalium:${EVAL_IMAGE_TAG}"
    IMAGE_TAGS+=("${GHCR_TAG}")
    PUSH_TAGS+=("${GHCR_TAG}")

    docker_login "${GHCR_REGISTRY}"
fi

if [[ ${#IMAGE_TAGS[@]} -eq 0 ]]; then
    log_fatal "No push targets configured (both PUSH_TO_HARBOR and PUSH_TO_GHCR are false)"
fi

# ---------------------------------------------------------------------------
# Build the evaluation image
# ---------------------------------------------------------------------------
declare -a BUILD_CMD=(
    docker build
    --file "${EVAL_DOCKERFILE}"
    --label "org.opencontainers.image.revision=${GIT_SHA}"
    --label "org.opencontainers.image.source=https://github.com/tenstorrent/tt-metal"
)

for tag in "${IMAGE_TAGS[@]}"; do
    BUILD_CMD+=(--tag "${tag}")
done

# Extra build-args from environment
if [[ -n "${BUILD_ARGS:-}" ]]; then
    for arg in ${BUILD_ARGS}; do
        BUILD_CMD+=(--build-arg "${arg}")
    done
fi

# ccache remote storage as a build secret
if [[ -n "${CCACHE_REMOTE_STORAGE:-}" ]]; then
    BUILD_CMD+=(--secret "id=CCACHE_REMOTE_STORAGE,env=CCACHE_REMOTE_STORAGE")
    export DOCKER_BUILDKIT=1
fi

[[ -n "${CACHE_FROM}" ]] && BUILD_CMD+=(--cache-from "${CACHE_FROM}")
[[ -n "${CACHE_TO}" ]] && BUILD_CMD+=(--cache-to "${CACHE_TO}")

BUILD_CMD+=(.)

log_info "Building evaluation image"
"${BUILD_CMD[@]}"

# ---------------------------------------------------------------------------
# Push to registries
# ---------------------------------------------------------------------------
for tag in "${PUSH_TAGS[@]}"; do
    log_info "Pushing: ${tag}"
    docker push "${tag}"
done

# ---------------------------------------------------------------------------
# Stage tags to shared storage
# ---------------------------------------------------------------------------
TAGS_FILE="$(mktemp)"
EVALUATION_URL="$(IFS=','; echo "${IMAGE_TAGS[*]}")"
echo "EVALUATION_IMAGE=${EVALUATION_URL}" > "${TAGS_FILE}"
stage_docker_tags "${PIPELINE_ID}" "${TAGS_FILE}"
rm -f "${TAGS_FILE}"

log_info "=== Evaluation image built and pushed ==="
for tag in "${IMAGE_TAGS[@]}"; do
    log_info "  ${tag}"
done
