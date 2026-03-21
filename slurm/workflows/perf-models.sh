#!/usr/bin/env bash
#SBATCH --job-name=perf-models
#SBATCH --partition=perf
#SBATCH --time=04:00:00
#
# GHA source: .github/workflows/perf-models-impl.yaml
# Runs model performance benchmarks across model-groups (llm_javelin, cnn_javelin,
# other_magic_env) on N300 WH-B0 and P150 BH.  Each array task corresponds to a
# model-group / arch combination defined in the matrix file.
#
# Environment overrides:
#   MATRIX_FILE  - JSON matrix mapping TASK_ID -> {name, cmd, arch, model_group, timeout}
#   MODEL        - Filter to a single model (default: all)
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
export INSTALL_WHEEL=1
setup_job
trap 'cleanup_job $?' EXIT

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODEL:-all}"
ARCH="${ARCH_NAME:-wormhole_b0}"

# ---------------------------------------------------------------------------
# Matrix-driven configuration
# ---------------------------------------------------------------------------
if [[ -n "${MATRIX_FILE:-}" ]]; then
    TEST_CMD="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "cmd")"
    TEST_NAME="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "name")"
    ARCH="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "arch" 2>/dev/null || echo "$ARCH")"
    MODEL_GROUP="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "model_group" 2>/dev/null || echo "")"
    TIMEOUT="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "timeout" 2>/dev/null || echo "45")"
    log_info "Running perf test: ${TEST_NAME} (${MODEL_GROUP}, ${ARCH})"
else
    # Embedded test catalog mirroring the GHA perf-models-impl matrix.
    # Each task runs all models for its model-group; the per-model tests
    # continue on failure (no -x) matching GHA's !cancelled() semantics.
    MODEL_GROUP="all"
    TIMEOUT=45
    TEST_NAME="perf-models-all"
    ARCH_DIR=$( [[ "$ARCH" == "blackhole" ]] && echo "blackhole" || echo "wormhole" )

    TEST_CMD="
        pytest models/demos/falcon7b_common/tests -m models_performance_bare_metal --timeout=1200 || true
        pytest models/demos/vision/segmentation/vanilla_unet/tests/ -m models_performance_bare_metal --timeout=600 || true
        pytest models/experimental/functional_unet/tests -m models_performance_bare_metal --timeout=480 || true
        pytest models/demos/vision/generative/stable_diffusion/${ARCH_DIR}/tests -m models_performance_bare_metal --timeout=600 || true
        pytest models/demos/vision/classification/resnet50/${ARCH_DIR}/tests/test_perf_e2e_resnet50.py -m models_performance_bare_metal --timeout=600 || true
        pytest models/demos/wormhole/bert_tiny/tests/test_performance.py -m models_performance_bare_metal --timeout=600 || true
        pytest models/demos/wormhole/distilbert/tests/test_perf_distilbert.py -m models_performance_bare_metal --timeout=600 || true
        pytest models/demos/vision/segmentation/segformer/tests/perf/ -m models_performance_bare_metal --timeout=600 || true
        pytest models/demos/metal_BERT_large_11/tests/ -m models_performance_bare_metal --timeout=600 || true
        pytest models/demos/vision/classification/mobilenetv2/tests/perf/ -m models_performance_bare_metal --timeout=600 || true
        pytest models/demos/vision/segmentation/vgg_unet/${ARCH_DIR}/tests/ -m models_performance_bare_metal --timeout=600 || true
        pytest models/experimental/swin_v2/tests/perf/ -m models_performance_bare_metal --timeout=600 || true
        pytest models/experimental/swin_s/tests/perf/ -m models_performance_bare_metal --timeout=600 || true
        pytest models/experimental/vovnet/tests/perf/ -m models_performance_bare_metal --timeout=600 || true
        pytest models/demos/vision/segmentation/ufld_v2/${ARCH_DIR}/tests/perf/ -m models_performance_bare_metal --timeout=600 || true
    "
fi

export ARCH_NAME="${ARCH}"

# ---------------------------------------------------------------------------
# Docker environment
# ---------------------------------------------------------------------------
HF_CACHE_PATH="${MLPERF_BASE}/huggingface/hub"

export DOCKER_EXTRA_VOLUMES="${MLPERF_BASE}:${MLPERF_BASE}:ro"
export DOCKER_EXTRA_ENV="GTEST_OUTPUT=xml:${TT_METAL_HOME}/generated/test_reports/
HF_HUB_CACHE=${HF_CACHE_PATH}
ARCH_NAME=${ARCH}"

# ---------------------------------------------------------------------------
# Run tests, merge perf reports, and stage benchmark data
# ---------------------------------------------------------------------------
docker_run "$DOCKER_IMAGE" "
    set -euo pipefail
    mkdir -p generated/test_reports generated/benchmark_data

    ${TEST_CMD}

    # Merge all generated perf reports into a single CSV
    python3 models/perf/merge_perf_results.py || true

    # Generate benchmark environment data if available
    if [ -d generated/benchmark_data ]; then
        python3 .github/scripts/data_analysis/create_benchmark_with_environment_json.py || true
    fi
"

# ---------------------------------------------------------------------------
# Stage artifacts
# ---------------------------------------------------------------------------
PERF_REPORT="Models_Perf_$(date +%Y_%m_%d).csv"
if [[ -f "${WORKSPACE}/${PERF_REPORT}" ]]; then
    log_info "Staging perf report: ${PERF_REPORT}"
    stage_test_report "${PIPELINE_ID}" "perf-models-${TASK_ID}" "${WORKSPACE}" || true
fi

log_info "Perf model tests complete (task ${TASK_ID})"
