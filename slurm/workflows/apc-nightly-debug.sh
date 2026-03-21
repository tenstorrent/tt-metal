#!/usr/bin/env bash
#SBATCH --job-name=apc-nightly-debug
#SBATCH --partition=wh-n150
#SBATCH --time=04:00:00
#
# Nightly APC (All Post-Commit) debug tests on Wormhole N150.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_lib setup_job
source_lib cleanup
source_config env

require_env PIPELINE_ID

log_info "=== APC nightly debug starting ==="
log_info "  Pipeline: ${PIPELINE_ID}"

BUILD_ARTIFACT=1 ENABLE_WATCHER=1 setup_job

IMAGE="$(resolve_image "${DOCKER_IMAGE:-}")"
docker_login
docker_pull_with_retry "${IMAGE}"

trap 'cleanup_job --exit-code $?' EXIT

DOCKER_EXTRA_ENV="PIPELINE_ID=${PIPELINE_ID}
TT_METAL_WATCHER=1
TT_METAL_WATCHER_APPEND=1
TT_METAL_WATCHER_NOINLINE=1
TT_METAL_SLOW_DISPATCH_MODE=1"
export DOCKER_EXTRA_ENV

docker_run "${IMAGE}" "
cd \${TT_METAL_HOME}
export PYTHONPATH=\${TT_METAL_HOME}

pytest tests/ -x --timeout=600 \
    -m 'not models_performance_bare_metal and not models_device_performance_bare_metal' \
    --junitxml=generated/test_reports/apc_nightly_debug.xml \
    -k 'not skip_in_debug' \
    2>&1 | tee generated/test_reports/apc_nightly_debug.log
"

log_info "=== APC nightly debug complete ==="
