#!/usr/bin/env bash
#SBATCH --job-name=fast-dispatch-build-and-unit-tests
#SBATCH --partition=wh-n150
#SBATCH --array=0-7
#SBATCH --time=02:00:00
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err

# Fast-dispatch unit tests: 8 array tasks mapping to 7 eager test splits + 1 sweep.
# Ports the matrix strategy from .github/workflows/fast-dispatch-build-and-unit-tests.yaml:
#   tasks 0-6  ->  pytest --splits 7 --group (TASK_ID+1)
#   task  7    ->  sweep tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/artifacts.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

parse_common_args "$@"
resolve_workflow_docker_image dev

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ARCH="${ARCH:-wormhole_b0}"
TEST_TIMEOUT="${TEST_TIMEOUT:-45}"          # per-job timeout in minutes (GHA default)
NUM_EAGER_SPLITS=7

export BUILD_ARTIFACT=1
export INSTALL_WHEEL=1
export ARCH_NAME="${ARCH}"
setup_job
trap 'cleanup_job $?' EXIT

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# ---------------------------------------------------------------------------
# Route array task to test group
# ---------------------------------------------------------------------------
if (( TASK_ID < NUM_EAGER_SPLITS )); then
    GROUP=$(( TASK_ID + 1 ))
    log_info "Running eager unit tests: split ${GROUP}/${NUM_EAGER_SPLITS} (task ${TASK_ID})"

    docker_run "$DOCKER_IMAGE" "
        set -euo pipefail
        export ARCH_NAME='${ARCH}'
        export LOGURU_LEVEL=INFO

        pytest tests/tt_eager/python_api_testing/unit_testing/ \
            -xvvv \
            --splits ${NUM_EAGER_SPLITS} \
            --group ${GROUP} \
            --splitting-algorithm least_duration \
            --timeout=600 \
            --junit-xml=generated/test_reports/eager_group_${GROUP}.xml
    "
else
    log_info "Running sweep tests (task ${TASK_ID})"

    docker_run "$DOCKER_IMAGE" "
        set -euo pipefail
        export ARCH_NAME='${ARCH}'
        export LOGURU_LEVEL=INFO

        pytest tests/tt_eager/python_api_testing/sweep_tests/pytests/ \
            -xvvv \
            --timeout=600 \
            --junit-xml=generated/test_reports/sweep.xml
    "
fi

log_info "Fast-dispatch unit tests complete (task ${TASK_ID})"
