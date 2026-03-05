#!/usr/bin/env bash
#SBATCH --job-name=fast-dispatch-frequent-tests
#SBATCH --partition=wh-n150
#SBATCH --time=02:00:00
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err
#
# GHA source: .github/workflows/fast-dispatch-frequent-tests-impl.yaml
# Each TASK_ID maps to a specific dispatch benchmark test group.

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

case "${TASK_ID}" in
    0)
        TEST_NAME="WH N300 pgm dispatch nightly"
        export ARCH_NAME=wormhole_b0
        TEST_CMD="\
            mkdir -p generated/test_reports && \
            ./build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch \
                --benchmark_out_format=json --benchmark_out=pgm_bench.json --benchmark_repetitions=2 && \
            ./tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/compare_pgm_dispatch_perf_ci.py pgm_bench.json"
        ;;
    1)
        TEST_NAME="WH N300 dispatch bandwidth IOMMU nightly"
        export ARCH_NAME=wormhole_b0
        TEST_CMD="\
            mkdir -p generated/test_reports && \
            ./build/test/tt_metal/perf_microbenchmark/dispatch/benchmark_rw_buffer \
                --benchmark_out_format=json --benchmark_out=buffer_bench.json --benchmark_repetitions=5 && \
            ./tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/compare_benchmark_rw_buffer.py buffer_bench.json \
                --ignore-times \
                --golden tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/benchmark_rw_buffer_n300_iommu_golden.json"
        ;;
    2)
        TEST_NAME="BH P150 pgm dispatch nightly"
        export ARCH_NAME=blackhole
        TEST_CMD="\
            mkdir -p generated/test_reports && \
            ./build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch \
                --benchmark_out_format=json --benchmark_out=pgm_bench.json --benchmark_repetitions=2 && \
            ./tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/compare_pgm_dispatch_perf_ci.py \
                -g tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/pgm_dispatch_blackhole_golden.json pgm_bench.json"
        ;;
    3)
        TEST_NAME="BH P150 dispatch bandwidth nightly"
        export ARCH_NAME=blackhole
        TEST_CMD="\
            mkdir -p generated/test_reports && \
            ./build/test/tt_metal/perf_microbenchmark/dispatch/benchmark_rw_buffer \
                --benchmark_out_format=json --benchmark_out=buffer_bench.json --benchmark_repetitions=11 && \
            ./tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/compare_benchmark_rw_buffer.py \
                --golden tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/benchmark_rw_buffer_blackhole_golden.json \
                buffer_bench.json --ignore-times"
        ;;
    *)
        log_fatal "Unknown task ID: ${TASK_ID} (expected 0-3)"
        ;;
esac

log_info "Running: ${TEST_NAME} (task ${TASK_ID})"

export DOCKER_EXTRA_ENV="GTEST_OUTPUT=xml:generated/test_reports/"
docker_run "$DOCKER_IMAGE" "$TEST_CMD"
