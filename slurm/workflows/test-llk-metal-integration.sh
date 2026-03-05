#!/usr/bin/env bash
#SBATCH --job-name=test-llk-metal-integration
#SBATCH --partition=wh-n150
#SBATCH --time=01:00:00
#
# LLK (Low-Level Kernel) integration tests with TT-Metal.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_lib setup_job
source_lib cleanup
source_config env

require_env PIPELINE_ID

log_info "=== LLK Metal integration tests starting ==="
log_info "  Pipeline: ${PIPELINE_ID}"

BUILD_ARTIFACT=1 setup_job

IMAGE="$(resolve_image "${DOCKER_IMAGE:-}")"
docker_login
docker_pull_with_retry "${IMAGE}"

trap 'cleanup_job --exit-code $?' EXIT

docker_run "${IMAGE}" "
cd \${TT_METAL_HOME}
export PYTHONPATH=\${TT_METAL_HOME}

./tests/scripts/run_llk_metal_integration.sh \
    2>&1 | tee generated/test_reports/llk_metal_integration.log

if [ -f tests/scripts/run_llk_unit_tests.sh ]; then
    ./tests/scripts/run_llk_unit_tests.sh \
        2>&1 | tee -a generated/test_reports/llk_metal_integration.log
fi
"

log_info "=== LLK Metal integration tests complete ==="
