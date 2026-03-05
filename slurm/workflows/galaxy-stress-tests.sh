#!/usr/bin/env bash
#SBATCH --job-name=galaxy-stress-tests
#SBATCH --partition=wh-galaxy
#SBATCH --time=06:00:00
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err

# Galaxy stress tests — array job with inline matrix covering the repeated
# decode stress test, fabric stability, and Llama long stress test.
# Equivalent to .github/workflows/galaxy-stress-tests-impl.yaml

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
# Inline matrix (mirrors GHA galaxy-stress + galaxy-long-tests jobs)
# ---------------------------------------------------------------------------
if [[ -z "${MATRIX_FILE:-}" ]]; then
    MATRIX_JSON='[
        {"name": "Galaxy decode stress (5x)",
         "timeout": 360,
         "cmd": "for i in {1..5}; do echo \"Run #$i\"; timeout --preserve-status 1200 pytest models/demos/llama3_70b_galaxy/demo/demo_decode.py -k \"nd-hang-test\"; done"},
        {"name": "Galaxy Fabric Stability Tests",
         "timeout": 100,
         "cmd": "TT_METAL_CLEAR_L1=1 build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_stability_6U_galaxy.yaml"},
        {"name": "Llama Galaxy Long Stress Test",
         "timeout": 140,
         "cmd": "LLAMA_DIR=/mnt/MLPerf/tt_dnn-models/llama/Llama3.3-70B-Instruct/ FAKE_DEVICE=TG pytest models/demos/llama3_70b_galaxy/demo/demo_decode.py -k \"stress-test and not mini-stress-test\""}
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
export DOCKER_EXTRA_ENV="LLAMA_DIR=/mnt/MLPerf/tt_dnn-models/llama/Llama3.3-70B-Instruct/
TT_METAL_ENABLE_ERISC_IRAM=1
LD_LIBRARY_PATH=/work/build/lib"
export DOCKER_EXTRA_VOLUMES="/mnt/MLPerf:/mnt/MLPerf:ro"

docker_run "$DOCKER_IMAGE" "
    ${TEST_CMD}
"

log_info "Galaxy stress test '${TEST_NAME}' complete"
