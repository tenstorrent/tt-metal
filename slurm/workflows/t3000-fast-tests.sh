#!/usr/bin/env bash
#SBATCH --job-name=t3000-fast-tests
#SBATCH --partition=wh-t3k
#SBATCH --time=01:30:00

# T3000 fast tests — array job with inline matrix (fabric + CCL tests).
# Equivalent to .github/workflows/t3000-fast-tests-impl.yaml

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
# Inline matrix (mirrors the GHA steps: fabric + ccl)
# ---------------------------------------------------------------------------
if [[ -z "${MATRIX_FILE:-}" ]]; then
    MATRIX_JSON='[
        {"name": "T3K fabric tests",   "cmd": "run_t3000_ttfabric_tests", "timeout": 20},
        {"name": "T3K CCL tests",      "cmd": "run_t3000_ccl_tests",      "timeout": 10}
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
export DOCKER_EXTRA_ENV="GTEST_OUTPUT=xml:${TT_METAL_HOME}/generated/test_reports/
LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib"
export DOCKER_EXTRA_OPTS="--memory 256g"

docker_run "$DOCKER_IMAGE" "
    mkdir -p \${TT_METAL_HOME}/generated/test_reports
    source tests/scripts/t3000/run_t3000_unit_tests.sh
    ${TEST_CMD}
"

log_info "T3000 fast test '${TEST_NAME}' complete"
