#!/usr/bin/env bash
#SBATCH --job-name=ttnn-post-commit
#SBATCH --partition=wh-n150
#SBATCH --time=02:00:00
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/artifacts.sh"
source "${SCRIPT_DIR}/lib/matrix.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

parse_common_args "$@"
resolve_docker_image dev

export BUILD_ARTIFACT=1
setup_job
trap 'cleanup_job $?' EXIT

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [[ -n "${MATRIX_FILE:-}" ]]; then
    TEST_GROUP="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "name")"
    TEST_CMD="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "cmd" || echo "")"
else
    TEST_GROUP="group_${TASK_ID}"
    TEST_CMD=""
fi

log_info "Running TTNN post-commit: ${TEST_GROUP} (task ${TASK_ID})"

if [[ -n "${TEST_CMD}" ]]; then
    docker_run "$DOCKER_IMAGE" "${TEST_CMD}"
else
    docker_run "$DOCKER_IMAGE" "\
        pytest tests/ttnn/unit_tests/ \
            -k '${TEST_GROUP}' \
            --timeout=600 \
            --junit-xml=generated/test_reports/ttnn_${TASK_ID}.xml
    "
fi
