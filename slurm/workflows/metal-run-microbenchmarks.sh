#!/usr/bin/env bash
#SBATCH --job-name=metal-run-microbenchmarks
#SBATCH --partition=perf
#SBATCH --time=02:00:00
#
# GHA source: .github/workflows/metal-run-microbenchmarks-impl.yaml
# Runs metal microbenchmarks including DRAM ubench, fabric BW/latency,
# data movement regressions, and PCIe performance.  Each array task maps
# to one benchmark group from the matrix file.
#
# Environment overrides:
#   MATRIX_FILE  - JSON matrix mapping TASK_ID -> {name, cmd, arch, ...}
#   ARCH_NAME    - Architecture override (default: from matrix or wormhole_b0)

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
    log_info "Running microbenchmark: ${TEST_NAME} (${ARCH})"
else
    # Embedded fallback catalog from GHA metal-run-microbenchmarks-impl.yaml
    case "${TASK_ID}" in
        0)
            TEST_NAME="N300 DRAM ubench"
            TEST_CMD="./tests/scripts/run_moreh_microbenchmark.sh"
            ARCH="wormhole_b0"
            ;;
        1)
            TEST_NAME="T3K ubench - Fabric BW & Latency"
            TEST_CMD='TT_METAL_CLEAR_L1=1 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_ubench_at_least_2x2_mesh.yaml'
            ARCH="wormhole_b0"
            ;;
        2)
            TEST_NAME="T3K ubench - Fabric Mux BW"
            TEST_CMD="pytest -svv tests/tt_metal/microbenchmarks/ethernet/test_fabric_mux_bandwidth.py"
            ARCH="wormhole_b0"
            ;;
        3)
            TEST_NAME="WH Data Movement Regressions"
            TEST_CMD="pytest -svv tests/tt_metal/tt_metal/data_movement/python/test_data_movement.py --gtest-filter Directed --verbose-log"
            ARCH="wormhole_b0"
            ;;
        4)
            TEST_NAME="BH P150 DRAM ubench"
            TEST_CMD="./tests/scripts/run_moreh_microbenchmark.sh"
            ARCH="blackhole"
            ;;
        5)
            TEST_NAME="BH Data Movement Regressions"
            TEST_CMD="pytest -svv tests/tt_metal/tt_metal/data_movement/python/test_data_movement.py --gtest-filter Directed --verbose-log"
            ARCH="blackhole"
            ;;
        6)
            TEST_NAME="BH P150 PCIe Performance"
            TEST_CMD='build/tools/mem_bench --benchmark_filter="Device (Reading|Writing) Host"'
            ARCH="blackhole"
            ;;
        7)
            TEST_NAME="Fabric Collectives ubench - Point to Point"
            TEST_CMD="tests/tt_metal/tt_fabric/benchmark/collectives/unicast/ci_run_unicast.sh"
            ARCH="wormhole_b0"
            ;;
        8)
            TEST_NAME="BH ubench - Fabric BW & Latency"
            TEST_CMD='TT_METAL_CLEAR_L1=1 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_ubench_at_least_2x2_mesh.yaml'
            ARCH="blackhole"
            ;;
        *)
            log_fatal "Unknown task ID: ${TASK_ID}"
            ;;
    esac
fi

export ARCH_NAME="${ARCH}"

# ---------------------------------------------------------------------------
# Docker environment — profiler is required for microbenchmarks
# ---------------------------------------------------------------------------
export DOCKER_EXTRA_ENV="TT_METAL_DEVICE_PROFILER=1
TRACY_NO_INVARIANT_CHECK=1
ARCH_NAME=${ARCH}"

log_info "Running microbenchmark: ${TEST_NAME} (task ${TASK_ID}, arch ${ARCH})"

# ---------------------------------------------------------------------------
# Run benchmark inside container
# ---------------------------------------------------------------------------
docker_run "$DOCKER_IMAGE" "
    set -euo pipefail
    mkdir -p generated/test_reports generated/profiler/.logs

    ${TEST_CMD}
"

# ---------------------------------------------------------------------------
# Stage profiler logs and benchmark data
# ---------------------------------------------------------------------------
PROFILER_DIR="${WORKSPACE}/generated/profiler/.logs"
if [[ -d "$PROFILER_DIR" ]]; then
    log_info "Staging profiler logs"
    stage_test_report "${PIPELINE_ID}" "microbenchmark-${TASK_ID}" "$PROFILER_DIR" || true
fi

log_info "Microbenchmark complete: ${TEST_NAME} (task ${TASK_ID})"
