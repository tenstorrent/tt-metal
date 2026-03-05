#!/usr/bin/env bash
#SBATCH --job-name=t3000-perf-tests
#SBATCH --partition=wh-t3k
#SBATCH --constraint=pipeline-perf
#SBATCH --time=03:00:00
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err

# T3000 performance tests — array job, dynamic matrix from
# tests/pipeline_reorg/t3k_perf_tests.yaml.
# Equivalent to .github/workflows/t3000-perf-tests-impl.yaml
# Runs on perf-constrained nodes with performance CPU governor.

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
# Enable performance CPU governor (restored in cleanup)
# ---------------------------------------------------------------------------
if command -v cpupower &>/dev/null; then
    sudo cpupower frequency-set -g performance 2>/dev/null || log_warn "Could not set CPU governor"
    register_cleanup 'sudo cpupower frequency-set -g ondemand 2>/dev/null || true'
fi

# ---------------------------------------------------------------------------
# Container execution
# ---------------------------------------------------------------------------
export DOCKER_EXTRA_ENV="GTEST_OUTPUT=xml:/work/generated/test_reports/
HF_HUB_CACHE=/mnt/MLPerf/huggingface/hub
GITHUB_ACTIONS=true
LD_LIBRARY_PATH=/work/build/lib"
export DOCKER_EXTRA_VOLUMES="/mnt/MLPerf:/mnt/MLPerf:ro"
export DOCKER_EXTRA_OPTS="--privileged -v /sys:/sys"

docker_run "$DOCKER_IMAGE" "
    mkdir -p /work/generated/test_reports
    source tests/scripts/t3000/run_t3000_perf_tests.sh
    echo '${TEST_CMD}'
    ${TEST_CMD}
    env python3 models/perf/merge_perf_results.py
"

log_info "T3000 perf test '${TEST_NAME}' complete"
