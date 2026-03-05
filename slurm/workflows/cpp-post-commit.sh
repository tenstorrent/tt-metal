#!/usr/bin/env bash
#SBATCH --job-name=cpp-post-commit
#SBATCH --partition=wh-n150
#SBATCH --time=02:00:00
#SBATCH --array=0-8
#
# GHA source: .github/workflows/cpp-post-commit.yaml
# Runs C++ unit tests. Each TASK_ID maps to a test group from the GHA matrix.
#
# Environment overrides:
#   NIGHTLY_RUN  - Set to 1 for nightly test set (default: 1, matching GHA nightly filter)
#   GTEST_FILTER - Custom gtest filter (default: "*")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/artifacts.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

parse_common_args "$@"
resolve_workflow_docker_image dev

export BUILD_ARTIFACT=1
setup_job
trap 'cleanup_job $?' EXIT

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
GTEST_FILTER="${GTEST_FILTER:-*}"

# Embedded test catalog from GHA cpp-post-commit.yaml (nightly set)
case "${TASK_ID}" in
    0)
        TEST_NAME="All C++"
        TEST_CMD="{ [ \"${GTEST_FILTER}\" = \"*NIGHTLY_*\" ]; } && \
            (echo 'Running unit_tests_legacy C++ nightly' && \
             ./build/test/tt_metal/unit_tests_legacy --gtest_filter=*NIGHTLY_*) && \
            ./tests/scripts/run_cpp_unit_tests.sh"
        ;;
    1)
        TEST_NAME="dispatch"
        TEST_CMD="TT_METAL_ENABLE_ERISC_IRAM=1 \
            ./build/test/tt_metal/unit_tests_dispatch --gtest_filter=*"
        ;;
    2)
        TEST_NAME="dispatch multicmd queue"
        TEST_CMD="TT_METAL_ENABLE_ERISC_IRAM=1 TT_METAL_GTEST_NUM_HW_CQS=2 \
            ./build/test/tt_metal/unit_tests_dispatch --gtest_filter=UnitMeshMultiCQ*Fixture.*"
        ;;
    3)
        TEST_NAME="distributed"
        TEST_CMD="./build/test/tt_metal/distributed/distributed_unit_tests --gtest_filter=*"
        ;;
    4)
        TEST_NAME="eth, misc, and user kernel path"
        TEST_CMD="./build/test/tt_metal/unit_tests_eth && \
            ./build/test/tt_metal/unit_tests_misc && \
            rm -rf /tmp/kernels && mkdir -p /tmp/kernels && \
            TT_METAL_KERNEL_PATH=/tmp/kernels ./build/test/tt_metal/unit_tests_api \
                --gtest_filter=CompileProgramWithKernelPathEnvVarFixture.*"
        ;;
    5)
        TEST_NAME="tools"
        TEST_CMD="./tests/scripts/run_tools_tests.sh && \
            ./build/test/tt_metal/unit_tests_debug_tools --gtest_filter=* && \
            ./build/test/tt_metal/unit_tests_inspector --gtest_filter=* && \
            TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1 \
                ./build/test/tt_metal/unit_tests_noc_debugging --gtest_filter=*"
        ;;
    6)
        TEST_NAME="ttnn tensor accessor"
        TEST_CMD="./build/test/ttnn/unit_tests_ttnn_accessor"
        ;;
    7)
        TEST_NAME="Nightly data movement"
        TEST_CMD="./build/test/tt_metal/unit_tests_data_movement --gtest_filter=${GTEST_FILTER}"
        ;;
    8)
        TEST_NAME="Nightly TTNN university lab examples"
        TEST_CMD="./build/test/ttnn/lab_examples/test_lab_eltwise_binary && \
            ./build/test/ttnn/lab_examples/test_lab_multicast"
        ;;
    *)
        log_fatal "Unknown task ID: ${TASK_ID} (expected 0-8)"
        ;;
esac

log_info "Running C++ test: ${TEST_NAME} (task ${TASK_ID})"

export DOCKER_EXTRA_ENV="GTEST_OUTPUT=xml:generated/test_reports/
LSAN_OPTIONS=suppressions=/usr/share/tt-metalium/lsan.supp"

docker_run "$DOCKER_IMAGE" "\
    mkdir -p generated/test_reports && \
    ${TEST_CMD}
"
