#!/usr/bin/env bash
#SBATCH --job-name=ttnn-run-sweeps
#SBATCH --partition=wh-n150
#SBATCH --time=12:00:00
#SBATCH --array=0-49%20
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err
#
# GHA source: .github/workflows/ttnn-run-sweeps.yaml
# Runs TTNN sweep framework tests in batched parallel execution.
#
# Environment overrides:
#   SWEEP_NAME       - Single sweep module or "ALL SWEEPS (Nightly)" etc.
#   SWEEP_MODE       - One of: nightly, comprehensive, model_traced, lead_models, single
#   SKIP_ON_TIMEOUT  - Set to 1 to skip remaining tests after first timeout
#   MEASURE_DEVICE_PERF - Set to 1 to enable device profiler
#   MEASURE_E2E_PERF - Set to 1 for cold/cached comparison
#   MEASURE_MEMORY   - Set to 1 for L1 memory measurement
#   MATRIX_FILE      - JSON matrix file mapping TASK_ID -> module_selector

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
resolve_workflow_docker_image dev

export BUILD_ARTIFACT=1
export INSTALL_WHEEL=1
setup_job
trap 'cleanup_job $?' EXIT

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Resolve sweep module from matrix file or direct env var
if [[ -n "${MATRIX_FILE:-}" ]]; then
    SWEEP_MODULE="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "module_selector")"
    SUITE_NAME="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "suite_name" 2>/dev/null || echo "")"
else
    SWEEP_MODULE="${SWEEP_NAME:-sweep_${TASK_ID}}"
    SUITE_NAME="${SUITE_NAME:-}"
fi
log_info "Running TTNN sweep: ${SWEEP_MODULE} (task ${TASK_ID})"

# Build runner command with optional flags
CMD=(
    python3 tests/sweep_framework/sweeps_runner.py
    --module-name "${SWEEP_MODULE}"
    --vector-source vectors_export
    --result-dest results_export
    --tag ci-main
    --summary
)

[[ "${SKIP_ON_TIMEOUT:-0}" == "1" ]] && CMD+=(--skip-on-timeout)
[[ "${MEASURE_DEVICE_PERF:-0}" == "1" ]] && CMD+=(--device-perf)
[[ "${MEASURE_E2E_PERF:-0}" == "1" ]] && CMD+=(--perf-with-cache)
[[ "${MEASURE_MEMORY:-0}" == "1" ]] && CMD+=(--measure-memory)
[[ -n "${SUITE_NAME}" ]] && CMD+=(--suite-name "${SUITE_NAME}")

export DOCKER_EXTRA_ENV="TRACY_NO_INVARIANT_CHECK=1
LOGURU_LEVEL=${LOGURU_LEVEL:-INFO}"
export DOCKER_EXTRA_VOLUMES="/mnt/MLPerf:/mnt/MLPerf:ro"

docker_run "$DOCKER_IMAGE" "${CMD[*]}"
