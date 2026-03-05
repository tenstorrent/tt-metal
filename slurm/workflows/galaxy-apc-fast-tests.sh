#!/usr/bin/env bash
#SBATCH --job-name=galaxy-apc-fast-tests
#SBATCH --partition=wh-galaxy
#SBATCH --time=01:30:00

# Galaxy APC fast tests — array job with inline matrix (Llama3 demo -k apc).
# Equivalent to .github/workflows/galaxy-apc-fast-tests-impl.yaml

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
# Inline matrix (single entry, but using array pattern for consistency)
# ---------------------------------------------------------------------------
if [[ -z "${MATRIX_FILE:-}" ]]; then
    MATRIX_JSON='[
        {"name": "Galaxy Llama3 demo tests",
         "cmd": "LLAMA_DIR=${MLPERF_BASE}/tt_dnn-models/llama/Llama3.3-70B-Instruct/ pytest models/demos/llama3_70b_galaxy/demo/text_demo.py -k apc",
         "timeout": 10}
    ]'
    MATRIX_FILE="$(create_matrix_file "$MATRIX_JSON")"
fi

TASK_ID="$(get_array_task_id)"
TEST_CMD="$(get_task_field "$MATRIX_FILE" "$TASK_ID" cmd)"
TEST_NAME="$(get_task_field "$MATRIX_FILE" "$TASK_ID" name)"

log_info "Running array task ${TASK_ID}: ${TEST_NAME}"

# ---------------------------------------------------------------------------
# Container execution
# ---------------------------------------------------------------------------
export DOCKER_EXTRA_ENV="LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib"
export DOCKER_EXTRA_VOLUMES="${MLPERF_BASE}:${MLPERF_BASE}:ro"

docker_run "$DOCKER_IMAGE" "
    ${TEST_CMD}
"

log_info "Galaxy APC fast test '${TEST_NAME}' complete"
