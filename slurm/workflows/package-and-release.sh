#!/usr/bin/env bash
#SBATCH --job-name=package-and-release
#SBATCH --partition=build
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err
#
# Package release artifacts: tarball, wheels, and release metadata.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_lib cleanup
source_config env

require_env PIPELINE_ID
require_env RELEASE_VERSION

log_info "=== Package and release starting ==="
log_info "  Pipeline: ${PIPELINE_ID}"
log_info "  Version:  ${RELEASE_VERSION}"

IMAGE="$(resolve_image "${DOCKER_IMAGE:-}")"
docker_login
docker_pull_with_retry "${IMAGE}"

DIST_DIR="${REPO_ROOT}/build/dist"
RELEASE_DIR="${ARTIFACT_DIR}/release/${RELEASE_VERSION}"
mkdir -p "${RELEASE_DIR}"

DOCKER_EXTRA_ENV="PIPELINE_ID=${PIPELINE_ID}
RELEASE_VERSION=${RELEASE_VERSION}
CMAKE_BUILD_PARALLEL_LEVEL=${SLURM_CPUS_PER_TASK:-8}"
export DOCKER_EXTRA_ENV

PACKAGE_COMMANDS="
cd /work

cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTT_METAL_BUILD_TESTS=OFF
cmake --build build --target install -- -j\${CMAKE_BUILD_PARALLEL_LEVEL:-8}

mkdir -p build/dist

tar --zstd -cf build/dist/tt-metal-\${RELEASE_VERSION}.tar.zst \
    --exclude='build/dist' \
    -C /work \
    build/lib build/bin build/generated

pip wheel . --no-deps --wheel-dir build/dist/
"

docker_run "${IMAGE}" "${PACKAGE_COMMANDS}"

log_info "Staging release artifacts to ${RELEASE_DIR}"
cp "${DIST_DIR}"/*.tar.zst "${RELEASE_DIR}/" 2>/dev/null || true
cp "${DIST_DIR}"/*.whl "${RELEASE_DIR}/" 2>/dev/null || true

cat > "${RELEASE_DIR}/release_meta.env" <<EOF
RELEASE_VERSION=${RELEASE_VERSION}
PIPELINE_ID=${PIPELINE_ID}
GIT_SHA=${GIT_SHA}
GIT_REF=${GIT_REF}
BUILT_AT=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
EOF

stage_build_artifact "${PIPELINE_ID}" "${DIST_DIR}"

WHEEL_FILE="$(ls "${DIST_DIR}"/*.whl 2>/dev/null | head -1)"
if [[ -n "${WHEEL_FILE}" ]]; then
    stage_wheel "${PIPELINE_ID}" "${WHEEL_FILE}"
fi

log_info "=== Package and release complete ==="
