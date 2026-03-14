#!/usr/bin/env bash
#SBATCH --job-name=run-profiler-regression
#SBATCH --partition=wh-n150
#SBATCH --time=02:00:00
#
# Tracy profiler regression tests.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_lib setup_job
source_lib cleanup
source_config env

require_env PIPELINE_ID

log_info "=== Profiler regression starting ==="
log_info "  Pipeline: ${PIPELINE_ID}"

BUILD_ARTIFACT=1 setup_job

IMAGE="$(resolve_image "${DOCKER_IMAGE:-}")"
docker_login
docker_pull_with_retry "${IMAGE}"

trap 'cleanup_job --exit-code $?' EXIT

TRACY_REPORT_DIR="${ARTIFACT_DIR}/reports/profiler-regression"
mkdir -p "${TRACY_REPORT_DIR}"

docker_run "${IMAGE}" "
cd \${TT_METAL_HOME}
export PYTHONPATH=\${TT_METAL_HOME}

python tests/tt_metal/tools/profiler/test_device_profiler.py \
    --regression \
    --output-dir generated/test_reports/profiler_regression \
    2>&1 | tee generated/test_reports/profiler_regression.log

if [ -d generated/test_reports/profiler_regression ]; then
    cp -r generated/test_reports/profiler_regression/* /artifacts/reports/profiler-regression/ 2>/dev/null || true
fi
"

log_info "=== Profiler regression complete ==="
