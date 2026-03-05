#!/usr/bin/env bash
#SBATCH --job-name=vllm-nightly-tests
#SBATCH --partition=wh-t3k
#SBATCH --time=04:00:00
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err
#
# vLLM nightly tests on T3K hardware.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_lib setup_job
source_lib cleanup
source_config env

require_env PIPELINE_ID

log_info "=== vLLM nightly tests starting ==="
log_info "  Pipeline: ${PIPELINE_ID}"

BUILD_ARTIFACT=1 INSTALL_WHEEL=1 setup_job

IMAGE="$(resolve_image "${DOCKER_IMAGE:-}")"
docker_login
docker_pull_with_retry "${IMAGE}"

trap 'cleanup_job --exit-code $?' EXIT

DOCKER_EXTRA_ENV="PIPELINE_ID=${PIPELINE_ID}
HF_HUB_CACHE=${HF_HUB_CACHE}
HF_HOME=${HF_HOME}
TT_CACHE_HOME=${TT_CACHE_HOME}
VLLM_TARGET_DEVICE=tt"
export DOCKER_EXTRA_ENV

DOCKER_EXTRA_VOLUMES="${MLPERF_BASE}:${MLPERF_BASE}:ro"
export DOCKER_EXTRA_VOLUMES

docker_run "${IMAGE}" "
cd /work
export PYTHONPATH=/work

pip install -e '.[vllm]' 2>/dev/null || pip install vllm

pytest tests/vllm/ \
    --timeout=1200 \
    --junitxml=generated/test_reports/vllm_nightly.xml \
    -v \
    2>&1 | tee generated/test_reports/vllm_nightly.log
"

log_info "=== vLLM nightly tests complete ==="
