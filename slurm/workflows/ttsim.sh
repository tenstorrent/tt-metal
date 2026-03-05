#!/usr/bin/env bash
#SBATCH --job-name=ttsim
#SBATCH --partition=build
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#
# TT simulator tests (no hardware required).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_lib cleanup
source_config env

require_env PIPELINE_ID

log_info "=== TTSim tests starting ==="
log_info "  Pipeline: ${PIPELINE_ID}"

IMAGE="$(resolve_image "${DOCKER_IMAGE:-}")"
docker_login
docker_pull_with_retry "${IMAGE}"

trap 'cleanup_job --exit-code $?' EXIT

DOCKER_EXTRA_ENV="PIPELINE_ID=${PIPELINE_ID}
ARCH_NAME=${ARCH_NAME:-wormhole_b0}
TT_METAL_SIMULATOR_EN=1"
export DOCKER_EXTRA_ENV

docker_run "${IMAGE}" "
cd \${TT_METAL_HOME}
export PYTHONPATH=\${TT_METAL_HOME}

cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DTT_METAL_BUILD_TESTS=ON \
    -DTT_METAL_SIMULATOR_EN=ON
cmake --build build --target install -- -j\${CMAKE_BUILD_PARALLEL_LEVEL:-8}

pytest tests/tt_metal/test_simulator.py \
    --timeout=600 \
    --junitxml=generated/test_reports/ttsim.xml \
    -v \
    2>&1 | tee generated/test_reports/ttsim.log
"

log_info "=== TTSim tests complete ==="
