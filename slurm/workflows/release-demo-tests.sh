#!/usr/bin/env bash
#SBATCH --job-name=release-demo-tests
#SBATCH --partition=wh-n150
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#
# Release validation tests run as a job array.
# Each task exercises a different demo/validation suite.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_lib setup_job
source_lib cleanup
source_config env

TASK_ID="$(get_array_task_id)"

declare -a TEST_SUITES=(
    "tests/scripts/run_demos.sh --release"
    "tests/scripts/run_ttnn_examples.sh"
    "tests/scripts/run_models_basic.sh --release"
    "tests/scripts/run_release_validation.sh"
)

if (( TASK_ID >= ${#TEST_SUITES[@]} )); then
    log_info "Task ${TASK_ID} exceeds suite count (${#TEST_SUITES[@]}), skipping"
    exit 0
fi

TEST_CMD="${TEST_SUITES[$TASK_ID]}"

log_info "=== Release demo tests starting ==="
log_info "  Task:    ${TASK_ID}/${#TEST_SUITES[@]}"
log_info "  Suite:   ${TEST_CMD}"

require_env PIPELINE_ID
BUILD_ARTIFACT=1 setup_job

IMAGE="$(resolve_image "${DOCKER_IMAGE:-}")"
docker_login
docker_pull_with_retry "${IMAGE}"

trap 'cleanup_job --exit-code $?' EXIT

docker_run "${IMAGE}" "
cd \${TT_METAL_HOME}
export PYTHONPATH=\${TT_METAL_HOME}
${TEST_CMD}
"

log_info "=== Release demo tests task ${TASK_ID} complete ==="
