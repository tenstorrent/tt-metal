#!/usr/bin/env bash
#SBATCH --job-name=galaxy-integration-tests
#SBATCH --partition=wh-galaxy
#SBATCH --time=06:00:00

# Galaxy integration tests — array job, dynamic matrix from
# tests/pipeline_reorg/galaxy_integration_tests.yaml.
# Equivalent to .github/workflows/galaxy-integration-tests-impl.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/artifacts.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/lib/matrix.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

parse_common_args "$@"
resolve_workflow_docker_image dev

export BUILD_ARTIFACT=1
export INSTALL_WHEEL=1
setup_job
trap 'cleanup_job $?' EXIT

# ---------------------------------------------------------------------------
# Matrix routing
# ---------------------------------------------------------------------------
TASK_ID="$(get_array_task_id)"
TEST_CMD="$(get_task_field "$MATRIX_FILE" "$TASK_ID" cmd)"
TEST_NAME="$(get_task_field "$MATRIX_FILE" "$TASK_ID" name)"
TEST_TIMEOUT="$(get_task_field "$MATRIX_FILE" "$TASK_ID" timeout)"

log_info "Running array task ${TASK_ID}: ${TEST_NAME}"

# ---------------------------------------------------------------------------
# Container execution
# ---------------------------------------------------------------------------
export DOCKER_EXTRA_ENV="LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib"
export DOCKER_EXTRA_VOLUMES="${MLPERF_BASE}:${MLPERF_BASE}:ro"

docker_run "$DOCKER_IMAGE" "
    echo '${TEST_CMD}'
    ${TEST_CMD}
"

log_info "Galaxy integration test '${TEST_NAME}' complete"
