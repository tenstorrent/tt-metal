#!/usr/bin/env bash
#SBATCH --job-name=perf-device-models
#SBATCH --partition=perf
#SBATCH --time=04:00:00
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err
#
# GHA source: .github/workflows/perf-device-models-impl.yaml
# Runs device-level performance regression tests for models across N300 WH-B0
# and P150 BH.  Each array task maps to a set of device perf tests from the
# matrix file (or falls back to the embedded catalog).
#
# Environment overrides:
#   MATRIX_FILE       - JSON matrix mapping TASK_ID -> {name, cmd, arch, ...}
#   REQUESTED_MODELS  - JSON array of model names, or "all" (default)
#   ARCH_NAME         - Architecture override (default: from matrix or wormhole_b0)

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
    DEVICE_PERF_MODEL="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "device_perf_model_name" 2>/dev/null || echo "")"
    log_info "Running device perf test: ${TEST_NAME} (${ARCH})"
else
    # Embedded fallback: run all device perf tests sequentially
    TIMEOUT=120
    TEST_NAME="device-perf-all"
    DEVICE_PERF_MODEL=""
    ARCH_DIR=$( [[ "$ARCH" == "blackhole" ]] && echo "blackhole" || echo "wormhole" )

    TEST_CMD="
        pytest models/demos/vision/generative/stable_diffusion/${ARCH_DIR}/tests -m models_device_performance_bare_metal --timeout=600 || true
        pytest models/demos/wormhole/distilbert/tests -m models_device_performance_bare_metal --timeout=600 || true
        pytest models/demos/vision/classification/vgg/tests/ -m models_device_performance_bare_metal --timeout=600 || true
        pytest models/demos/vision/segmentation/vgg_unet/${ARCH_DIR}/tests/perf -m models_device_performance_bare_metal --timeout=600 || true
        pytest models/demos/wormhole/bert_tiny/tests -m models_device_performance_bare_metal --timeout=600 || true
        pytest models/demos/squeezebert/tests -m models_device_performance_bare_metal --timeout=600 || true
        pytest models/demos/vision/classification/resnet50/${ARCH_DIR}/tests -m models_device_performance_bare_metal --timeout=600 || true
        pytest models/demos/vision/segmentation/ufld_v2/${ARCH_DIR}/tests/perf -m models_device_performance_bare_metal --timeout=600 || true
        pytest models/demos/${ARCH_DIR}/sentence_bert/tests/perf -m models_device_performance_bare_metal --timeout=600 || true
        pytest models/experimental/functional_unet/tests/test_unet_perf.py -m models_device_performance_bare_metal --timeout=600 || true
        pytest models/demos/wormhole/mamba/tests -m models_device_performance_bare_metal --timeout=600 || true
        pytest models/demos/vision/classification/mobilenetv2/tests/perf -m models_device_performance_bare_metal --timeout=600 || true
        pytest models/demos/vision/classification/vit/${ARCH_DIR}/tests/ -m models_device_performance_bare_metal --timeout=600 || true
        pytest models/demos/vision/segmentation/vanilla_unet/tests -m models_device_performance_bare_metal --timeout=600 || true
        pytest models/experimental/swin_v2/tests/perf -m models_device_performance_bare_metal --timeout=600 || true
        pytest models/experimental/swin_s/tests/perf -m models_device_performance_bare_metal --timeout=600 || true
        pytest models/experimental/vovnet/tests/perf -m models_device_performance_bare_metal --timeout=600 || true
        pytest models/experimental/efficientnetb0/tests/perf -m models_device_performance_bare_metal --timeout=600 || true
        pytest tests/ttnn/perf_tests/operations/conv -m models_device_performance_bare_metal --timeout=300 || true
    "
fi

export ARCH_NAME="${ARCH}"

# ---------------------------------------------------------------------------
# Docker environment
# ---------------------------------------------------------------------------
export DOCKER_EXTRA_VOLUMES="/mnt/MLPerf:/mnt/MLPerf:ro"
export DOCKER_EXTRA_ENV="GTEST_OUTPUT=xml:/work/generated/test_reports/
TRACY_NO_INVARIANT_CHECK=1
ARCH_NAME=${ARCH}"

# ---------------------------------------------------------------------------
# Run tests, generate device perf report, stage benchmark data
# ---------------------------------------------------------------------------
docker_run "$DOCKER_IMAGE" "
    set -euo pipefail
    mkdir -p generated/test_reports generated/benchmark_data

    ${TEST_CMD}

    # Merge device perf results into a dated CSV
    DEVICE_PERF_REPORT_FILENAME=\"Models_Device_Perf_\$(date +%Y_%m_%d).csv\"
    if [ -f \"\$DEVICE_PERF_REPORT_FILENAME\" ]; then
        python3 models/perf/merge_device_perf_results.py \"\$DEVICE_PERF_REPORT_FILENAME\" CHECK || true
    fi

    # Generate benchmark environment data if available
    if [ -d generated/benchmark_data ]; then
        python3 .github/scripts/data_analysis/create_benchmark_with_environment_json.py || true
    fi
"

log_info "Device perf model tests complete (task ${TASK_ID})"
