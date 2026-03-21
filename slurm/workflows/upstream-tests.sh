#!/usr/bin/env bash
#SBATCH --job-name=upstream-tests
#SBATCH --partition=wh-n150
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#
# Upstream integration tests run as a job array.
# Matrix configuration loaded from config/upstream-tests.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_lib matrix
source_lib setup_job
source_lib cleanup
source_config env

require_env PIPELINE_ID

TASK_ID="$(get_array_task_id)"
MATRIX_FILE="${MATRIX_FILE:-${SCRIPT_DIR}/../config/upstream-tests.json}"

log_info "=== Upstream tests starting ==="
log_info "  Pipeline: ${PIPELINE_ID}"
log_info "  Task:     ${TASK_ID}"
log_info "  Matrix:   ${MATRIX_FILE}"

if [[ ! -f "${MATRIX_FILE}" ]]; then
    log_fatal "Matrix file not found: ${MATRIX_FILE}"
fi

TASK_CONFIG="$(get_task_config "${MATRIX_FILE}" "${TASK_ID}" 2>/dev/null || echo '{}')"
TEST_NAME="$(echo "${TASK_CONFIG}" | jq -r '.name // "unknown"')"
TEST_CMD="$(echo "${TASK_CONFIG}" | jq -r '.command // ""')"
TEST_TIMEOUT="$(echo "${TASK_CONFIG}" | jq -r '.timeout // "600"')"

if [[ -z "${TEST_CMD}" || "${TEST_CMD}" == "null" ]]; then
    log_info "No command for task ${TASK_ID}, skipping"
    exit 0
fi

log_info "  Test:     ${TEST_NAME}"
log_info "  Command:  ${TEST_CMD}"

BUILD_ARTIFACT=1 setup_job

IMAGE="$(resolve_image "${DOCKER_IMAGE:-}")"
docker_login
docker_pull_with_retry "${IMAGE}"

trap 'cleanup_job --exit-code $?' EXIT

docker_run "${IMAGE}" "
cd \${TT_METAL_HOME}
export PYTHONPATH=\${TT_METAL_HOME}
timeout ${TEST_TIMEOUT} ${TEST_CMD} \
    2>&1 | tee generated/test_reports/upstream_${TASK_ID}.log
"

log_info "=== Upstream tests task ${TASK_ID} (${TEST_NAME}) complete ==="
