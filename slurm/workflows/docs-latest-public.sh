#!/usr/bin/env bash
#SBATCH --job-name=docs-latest-public
#SBATCH --partition=build
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err
#
# Build and publish public documentation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_config env

require_env PIPELINE_ID

DOCS_OUTPUT="${ARTIFACT_DIR}/docs"
mkdir -p "${DOCS_OUTPUT}"

log_info "=== Docs build starting ==="
log_info "  Pipeline: ${PIPELINE_ID}"

IMAGE="$(resolve_image "${DOCKER_IMAGE:-}")"
docker_login
docker_pull_with_retry "${IMAGE}"

DOCKER_EXTRA_ENV="PIPELINE_ID=${PIPELINE_ID}
DOCS_VERSION=${DOCS_VERSION:-latest}"
export DOCKER_EXTRA_ENV

BUILD_COMMANDS="
cd /work

pip install -r docs/requirements-docs.txt 2>/dev/null || true

cd docs
make clean
make html SPHINXOPTS='-W --keep-going'

cp -r build/html /artifacts/docs/
"

docker_run "${IMAGE}" "${BUILD_COMMANDS}"

log_info "Documentation built: ${DOCS_OUTPUT}/docs/"

if [[ "${PUBLISH_DOCS:-0}" == "1" ]]; then
    log_info "Publishing docs (PUBLISH_DOCS=1)"
    DOCS_DEST="${DOCS_PUBLISH_PATH:-/weka/ci/docs/latest}"
    mkdir -p "${DOCS_DEST}"
    rsync -a --delete "${DOCS_OUTPUT}/docs/" "${DOCS_DEST}/"
    log_info "Docs published to ${DOCS_DEST}"
fi

log_info "=== Docs build complete ==="
