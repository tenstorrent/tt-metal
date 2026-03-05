#!/usr/bin/env bash
#SBATCH --job-name=ops-post-commit
#SBATCH --partition=wh-n150
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#
# GHA source: .github/workflows/ops-post-commit.yaml
# Runs ops unit tests. Each TASK_ID maps to a test group defined in the
# embedded catalog (mirrors the GHA define-ops-tests matrix).
#
# Environment overrides:
#   MATRIX_FILE  - JSON matrix mapping TASK_ID -> {name, cmd, arch, timeout}
#   ARCH_NAME    - Filter tests by architecture (default: wormhole_b0)

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
ARCH="${ARCH_NAME:-wormhole_b0}"

if [[ -n "${MATRIX_FILE:-}" ]]; then
    TEST_NAME="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "name")"
    TEST_CMD="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "cmd")"
    TIMEOUT="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "timeout" 2>/dev/null || echo "20")"
    SLOW_DISPATCH="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "slow_dispatch" 2>/dev/null || echo "false")"
else
    # Embedded test catalog from GHA ops-post-commit.yaml
    case "${TASK_ID}" in
        0)
            TEST_NAME="wormhole_b0 sdxl op tests"
            TEST_CMD='pytest --timeout 300 models/experimental/stable_diffusion_xl_base/tests/test_sdxl_op_unit_test_perf.py -k "_performance" -xv --tb=short'
            TIMEOUT=5
            SLOW_DISPATCH="false"
            [[ "${ARCH}" != "wormhole_b0" ]] && { log_info "Skipping WH-only test on ${ARCH}"; exit 0; }
            ;;
        1)
            TEST_NAME="blackhole deepseek blitz op tests (slow dispatch)"
            TEST_CMD='pytest --timeout 600 -m "not skip_post_commit" models/demos/deepseek_v3_b1/tests/unit_tests/'
            TIMEOUT=20
            SLOW_DISPATCH="true"
            [[ "${ARCH}" != "blackhole" ]] && { log_info "Skipping BH-only test on ${ARCH}"; exit 0; }
            ;;
        2)
            TEST_NAME="blackhole deepseek blitz op tests (fast dispatch)"
            TEST_CMD='pytest --timeout 600 -m "not skip_post_commit" models/demos/deepseek_v3_b1/tests/unit_tests/'
            TIMEOUT=20
            SLOW_DISPATCH="false"
            [[ "${ARCH}" != "blackhole" ]] && { log_info "Skipping BH-only test on ${ARCH}"; exit 0; }
            ;;
        *)
            log_fatal "Unknown task ID: ${TASK_ID}"
            ;;
    esac
fi

log_info "Running ops test: ${TEST_NAME} (task ${TASK_ID}, timeout ${TIMEOUT}m)"

EXTRA_ENV="TRACY_NO_INVARIANT_CHECK=1"
if [[ "${SLOW_DISPATCH}" == "true" ]]; then
    EXTRA_ENV="${EXTRA_ENV}
TT_METAL_SLOW_DISPATCH_MODE=1"
fi
export DOCKER_EXTRA_ENV="${EXTRA_ENV}"

docker_run "$DOCKER_IMAGE" "\
    ${TEST_CMD} && \
    export DEVICE_PERF_REPORT_FILENAME=Ops_Perf.csv && \
    python3 models/perf/merge_device_perf_results.py \"\$DEVICE_PERF_REPORT_FILENAME\" REPORT && \
    cat \"\$DEVICE_PERF_REPORT_FILENAME\"
"
