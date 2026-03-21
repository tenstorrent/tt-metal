#!/usr/bin/env bash
#SBATCH --job-name=blackhole-multi-card-unit-tests
#SBATCH --partition=bh-p300
#SBATCH --time=02:00:00

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/artifacts.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/lib/matrix.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

export ARCH_NAME=blackhole
export BUILD_ARTIFACT=1

parse_common_args "$@"
resolve_docker_image dev
setup_job
trap 'cleanup_job --exit-code $?' EXIT

if [[ -n "${MATRIX_FILE:-}" ]]; then
    TEST_CMD="$(get_task_field cmd)"
    TEST_NAME="$(get_task_field name)"
    log_info "Running test: ${TEST_NAME}"
else
    TEST_CMD="pytest tests/tt_metal/blackhole/multi_card_unit -x --timeout=600"
fi

docker_run "$DOCKER_IMAGE" "${TEST_CMD}"
