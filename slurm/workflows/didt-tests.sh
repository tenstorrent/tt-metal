#!/usr/bin/env bash
#SBATCH --job-name=didt-tests
#SBATCH --partition=wh-n150
#SBATCH --time=02:00:00
#
# GHA source: .github/workflows/didt-tests.yaml
# Runs di/dt (power) stress tests across multiple workloads with varying
# TT_MM_THROTTLE_PERF settings.  Tests verify determinism under power stress.
#
# Environment overrides:
#   ARCH_NAME             - Architecture (default: wormhole_b0)
#   DIDT_ITERATIONS       - Workload iterations per test (default: 100)
#   DIDT_CHECK_INTERVAL   - Determinism check interval (default: 1)

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
export INSTALL_WHEEL=1
setup_job
trap 'cleanup_job $?' EXIT

ARCH="${ARCH_NAME:-wormhole_b0}"
ITERATIONS="${DIDT_ITERATIONS:-100}"
CHECK_INTERVAL="${DIDT_CHECK_INTERVAL:-1}"

export ARCH_NAME="${ARCH}"

log_info "Running DIDT tests (arch=${ARCH}, iterations=${ITERATIONS}, check_interval=${CHECK_INTERVAL})"

# ---------------------------------------------------------------------------
# Docker environment
# ---------------------------------------------------------------------------
export DOCKER_EXTRA_ENV="ARCH_NAME=${ARCH}"

# ---------------------------------------------------------------------------
# Run all DIDT workloads — mirrors GHA didt-tests.yaml step "Run di/dt Tests"
# Each test runs with a specific TT_MM_THROTTLE_PERF value.
# ---------------------------------------------------------------------------
docker_run "$DOCKER_IMAGE" "
    set -euo pipefail

    TT_MM_THROTTLE_PERF=1 pytest tests/didt/test_resnet_conv.py::test_resnet_conv \
        -k 'all' --didt-workload-iterations ${ITERATIONS} --determinism-check-interval ${CHECK_INTERVAL}

    TT_MM_THROTTLE_PERF=0 pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul \
        -k 'without_gelu and all' --didt-workload-iterations ${ITERATIONS} --determinism-check-interval ${CHECK_INTERVAL}

    TT_MM_THROTTLE_PERF=0 pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul \
        -k 'with_gelu and all' --didt-workload-iterations ${ITERATIONS} --determinism-check-interval ${CHECK_INTERVAL}

    TT_MM_THROTTLE_PERF=1 pytest tests/didt/test_lm_head_matmul.py::test_lm_head_matmul \
        -k 'all' --didt-workload-iterations ${ITERATIONS} --determinism-check-interval ${CHECK_INTERVAL}

    TT_MM_THROTTLE_PERF=5 pytest tests/didt/test_sdxl_conv.py::test_sdxl_conv \
        -k 'all' --didt-workload-iterations ${ITERATIONS} --determinism-check-interval ${CHECK_INTERVAL}

    TT_MM_THROTTLE_PERF=5 pytest tests/didt/test_sdxl_matmul.py::test_sdxl_matmul \
        -k 'all' --didt-workload-iterations ${ITERATIONS} --determinism-check-interval ${CHECK_INTERVAL}

    TT_MM_THROTTLE_PERF=5 pytest tests/didt/test_sdxl_conv_1280x1280_upsample.py::test_sdxl_conv \
        -k 'all' --didt-workload-iterations ${ITERATIONS} --determinism-check-interval ${CHECK_INTERVAL}

    pytest tests/didt/test_mla_sdpa.py::test_mla_sdpa \
        -k 'all' --didt-workload-iterations 10 --determinism-check-interval ${CHECK_INTERVAL}

    TT_MM_THROTTLE_PERF=1 pytest tests/didt/test_deepseek_v3_128k_matmul.py \
        -k '1chips or galaxy' --didt-workload-iterations 10 --determinism-check-interval ${CHECK_INTERVAL}
"

log_info "DIDT tests complete (arch=${ARCH})"
