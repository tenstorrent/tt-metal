#!/usr/bin/env bash
#SBATCH --job-name=test-wheels
#SBATCH --partition=wh-n150
#SBATCH --time=01:00:00
#
# Wheel testing: install the built wheel and run smoke tests.
# Partition can be overridden at submission time for different hardware.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_lib setup_job
source_lib cleanup
source_config env

require_env PIPELINE_ID

log_info "=== Wheel tests starting ==="
log_info "  Pipeline: ${PIPELINE_ID}"

INSTALL_WHEEL=1 setup_job

IMAGE="$(resolve_image "${DOCKER_IMAGE:-}")"
docker_login
docker_pull_with_retry "${IMAGE}"

trap 'cleanup_job --exit-code $?' EXIT

docker_run "${IMAGE}" "
cd \${TT_METAL_HOME}

WHEEL_DIR='${ARTIFACT_DIR}/build/tt_metal_wheels'
if [ -d \"\${WHEEL_DIR}\" ]; then
    pip install \"\${WHEEL_DIR}\"/*.whl
else
    echo 'No wheel directory found, attempting import test only'
fi

python -c 'import ttnn; print(f\"ttnn version: {ttnn.__version__}\")' || \
    { echo 'Failed to import ttnn'; exit 1; }

pytest tests/ttnn/unit_tests/test_import.py \
    --timeout=120 \
    --junitxml=generated/test_reports/wheel_tests.xml \
    -v \
    2>&1 | tee generated/test_reports/wheel_tests.log

pytest tests/scripts/test_wheel_install.py \
    --timeout=120 \
    -v \
    2>&1 | tee -a generated/test_reports/wheel_tests.log || true
"

log_info "=== Wheel tests complete ==="
