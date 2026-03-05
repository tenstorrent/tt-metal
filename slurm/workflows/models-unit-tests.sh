#!/usr/bin/env bash
#SBATCH --job-name=models-unit-tests
#SBATCH --time=02:00:00
#
# GHA source: .github/workflows/models-unit-tests-impl.yaml
# Worker: runs model unit tests from a matrix-defined command.
# Partition is set dynamically by the orchestrator (models-t{1,2,3}-unit-tests.sh)
# via the matrix config's partition field.
#
# The matrix is loaded from tests/pipeline_reorg/models_unit_tests.yaml by the
# orchestrator and each array element has: name, cmd, timeout, runs_on, tier, owner_id.
#
# Environment overrides:
#   MATRIX_FILE       - JSON matrix mapping TASK_ID -> {name, cmd, timeout, ...}
#   ARCH_NAME         - Architecture override
#   MLPERF_READ_ONLY  - Mount MLPerf read-only (default: true)

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
MLPERF_MODE="${MLPERF_READ_ONLY:-true}"
MLPERF_OPTS=$( [[ "$MLPERF_MODE" == "true" ]] && echo ":ro" || echo ":rw" )

# ---------------------------------------------------------------------------
# Matrix-driven configuration
# ---------------------------------------------------------------------------
if [[ -n "${MATRIX_FILE:-}" ]]; then
    TEST_CMD="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "cmd")"
    TEST_NAME="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "name")"
    TIMEOUT="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "timeout" 2>/dev/null || echo "60")"
    log_info "Running model unit test: ${TEST_NAME} (task ${TASK_ID}, timeout ${TIMEOUT}m)"
else
    TEST_NAME="models-unit-tests-fallback"
    TIMEOUT=60
    TEST_CMD="source tests/scripts/run_python_model_tests.sh && run_python_model_tests_${ARCH} && run_python_model_tests_slow_runtime_mode_${ARCH}"
fi

export ARCH_NAME="${ARCH}"

# ---------------------------------------------------------------------------
# Docker environment — mirrors GHA container env block
# ---------------------------------------------------------------------------
export DOCKER_EXTRA_VOLUMES="${MLPERF_BASE}:${MLPERF_BASE}${MLPERF_OPTS}"
export DOCKER_EXTRA_ENV="HF_HUB_OFFLINE=1
HF_HUB_CACHE=${MLPERF_BASE}/huggingface/hub
GTEST_OUTPUT=xml:${TT_METAL_HOME}/generated/test_reports/
ARCH_NAME=${ARCH}"
export DOCKER_EXTRA_OPTS="--privileged -v /sys:/sys"

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
docker_run "$DOCKER_IMAGE" "
    set -euo pipefail
    mkdir -p generated/test_reports

    ${TEST_CMD}
"

log_info "Model unit tests complete: ${TEST_NAME} (task ${TASK_ID})"
