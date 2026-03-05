#!/usr/bin/env bash
#SBATCH --job-name=temp-ccache-test
#SBATCH --partition=build
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err
#
# Ccache testing: validate kernel ccache hit rates and remote storage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib docker
source_lib setup_job
source_lib cleanup
source_config env

require_env PIPELINE_ID

log_info "=== Ccache test starting ==="
log_info "  Pipeline: ${PIPELINE_ID}"

ENABLE_KERNEL_CCACHE=1 setup_job

IMAGE="$(resolve_image "${DOCKER_IMAGE:-}")"
docker_login
docker_pull_with_retry "${IMAGE}"

trap 'cleanup_job --exit-code $?' EXIT

DOCKER_EXTRA_ENV="PIPELINE_ID=${PIPELINE_ID}
CMAKE_BUILD_PARALLEL_LEVEL=${SLURM_CPUS_PER_TASK:-8}
CCACHE_COMPILERCHECK=content
CCACHE_REMOTE_ONLY=true
CCACHE_NOHASHDIR=true
TT_METAL_CCACHE_KERNEL_SUPPORT=1
CCACHE_REMOTE_STORAGE=${CCACHE_REMOTE_STORAGE:-}"
export DOCKER_EXTRA_ENV

docker_run "${IMAGE}" "
cd /work

ccache -z
ccache -p

cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DTT_METAL_BUILD_TESTS=OFF
cmake --build build --target install -- -j\${CMAKE_BUILD_PARALLEL_LEVEL:-8}

echo '=== Ccache stats ==='
ccache -s
echo '=== Ccache stats end ==='

HITS=\$(ccache -s | grep 'cache hit rate' | grep -o '[0-9.]*' | head -1 || echo '0')
echo \"Cache hit rate: \${HITS}%\"
"

log_info "=== Ccache test complete ==="
