#!/usr/bin/env bash
#SBATCH --job-name=profiler-tests
#SBATCH --partition=wh-n150
#SBATCH --time=02:00:00
#
# Profiler unit and integration tests.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_lib setup_job
source_lib cleanup
source_config env

require_env PIPELINE_ID

log_info "=== Profiler tests starting ==="
log_info "  Pipeline: ${PIPELINE_ID}"

BUILD_ARTIFACT=1 setup_job

IMAGE="$(resolve_image "${DOCKER_IMAGE:-}")"
docker_login
docker_pull_with_retry "${IMAGE}"

trap 'cleanup_job --exit-code $?' EXIT

docker_run "${IMAGE}" "
cd \${TT_METAL_HOME}
export PYTHONPATH=\${TT_METAL_HOME}

pytest tests/tt_metal/tools/profiler/ \
    --timeout=600 \
    --junitxml=generated/test_reports/profiler_tests.xml \
    -v \
    2>&1 | tee generated/test_reports/profiler_tests.log

./tests/scripts/run_profiler_regressions.sh 2>&1 | tee -a generated/test_reports/profiler_tests.log || true
"

log_info "=== Profiler tests complete ==="
