#!/usr/bin/env bash
#SBATCH --job-name=wheels-test
#SBATCH --time=02:00:00
#
# Test a built wheel on real hardware. Fetches the wheel artifact from
# shared storage, installs it, and runs end-to-end tests.
#
# Submitted by build-and-test-wheels.sh with --dependency=afterok on the
# wheel build job.
#
# Environment:
#   PIPELINE_ID       (required)
#   PYTHON_VERSION    Python version matching the built wheel (default: 3.10)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_lib setup_job
source_lib cleanup
source_config env

require_env PIPELINE_ID

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

log_info "=== Wheel test runner starting ==="
log_info "  Pipeline: ${PIPELINE_ID}"
log_info "  Python:   ${PYTHON_VERSION}"

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
    echo 'ERROR: No wheel directory found at \${WHEEL_DIR}'
    exit 1
fi

python -c 'import ttnn; print(f\"ttnn version: {ttnn.__version__}\")' || \
    { echo 'Failed to import ttnn'; exit 1; }

# Host-side end-to-end tests
if [ -f tests/scripts/set_up_end_to_end_tests_env.sh ]; then
    ./tests/scripts/set_up_end_to_end_tests_env.sh
    source tests/end_to_end_tests/env/bin/activate
    cd tests/end_to_end_tests
    pytest -c conftest.py . -m eager_host_side \
        --timeout=120 \
        --junitxml=\${TT_METAL_HOME}/generated/test_reports/wheel_e2e_tests.xml \
        -v \
        2>&1 | tee \${TT_METAL_HOME}/generated/test_reports/wheel_e2e_tests.log
    cd \${TT_METAL_HOME}
fi

# Basic import and unit tests
pytest tests/ttnn/unit_tests/test_import.py \
    --timeout=120 \
    --junitxml=generated/test_reports/wheel_import_tests.xml \
    -v \
    2>&1 | tee generated/test_reports/wheel_import_tests.log

pytest tests/scripts/test_wheel_install.py \
    --timeout=120 \
    -v \
    2>&1 | tee -a generated/test_reports/wheel_import_tests.log || true
"

log_info "=== Wheel test runner complete ==="
