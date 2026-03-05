#!/usr/bin/env bash
#SBATCH --job-name=tg-op-perf-tests
#SBATCH --partition=wh-galaxy
#SBATCH --time=05:00:00
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err
#
# GHA source: .github/workflows/galaxy-perf-tests.yaml (TG op perf subset)
# Runs TG (Galaxy) operator performance tests on a multi-chip Galaxy system.
#
# Environment overrides:
#   MATRIX_FILE  - JSON matrix mapping TASK_ID -> {name, cmd, arch, timeout}
#   ARCH_NAME    - Architecture override (default: wormhole_b0)

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

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARCH="${ARCH_NAME:-wormhole_b0}"

# ---------------------------------------------------------------------------
# Matrix-driven configuration
# ---------------------------------------------------------------------------
if [[ -n "${MATRIX_FILE:-}" ]]; then
    TEST_CMD="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "cmd")"
    TEST_NAME="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "name")"
    ARCH="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "arch" 2>/dev/null || echo "$ARCH")"
    TIMEOUT="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "timeout" 2>/dev/null || echo "120")"
    log_info "Running TG op perf test: ${TEST_NAME}"
else
    TEST_NAME="tg-op-perf-all"
    TIMEOUT=120
    TEST_CMD="pytest tests/ttnn/unit_tests/operations/test_all_gather.py -m models_performance_bare_metal --timeout=1200"
fi

export ARCH_NAME="${ARCH}"

# ---------------------------------------------------------------------------
# Docker environment
# ---------------------------------------------------------------------------
export DOCKER_EXTRA_VOLUMES="/mnt/MLPerf:/mnt/MLPerf:ro"
export DOCKER_EXTRA_ENV="GTEST_OUTPUT=xml:/work/generated/test_reports/
TRACY_NO_INVARIANT_CHECK=1
ARCH_NAME=${ARCH}"

log_info "Running TG op perf: ${TEST_NAME} (task ${TASK_ID}, arch ${ARCH})"

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
docker_run "$DOCKER_IMAGE" "
    set -euo pipefail
    mkdir -p generated/test_reports

    ${TEST_CMD}
"

log_info "TG op perf tests complete (task ${TASK_ID})"
