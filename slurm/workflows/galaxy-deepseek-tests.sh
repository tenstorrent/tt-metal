#!/usr/bin/env bash
#SBATCH --job-name=galaxy-deepseek-tests
#SBATCH --partition=wh-galaxy
#SBATCH --time=03:00:00

# Galaxy DeepSeek tests — array job with inline matrix (unit, module,
# long-seq-module).
# Equivalent to .github/workflows/galaxy-deepseek-tests-impl.yaml

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
# Inline matrix (mirrors GHA generate-matrix)
# ---------------------------------------------------------------------------
if [[ -z "${MATRIX_FILE:-}" ]]; then
    MATRIX_JSON='[
        {"name": "(Galaxy) DeepSeek unit tests",
         "test_type": "unit",
         "timeout": 15,
         "cmd": "uv pip install -r models/demos/deepseek_v3/reference/deepseek/requirements.txt && pytest models/demos/deepseek_v3/tests/unit --timeout 60 --durations=0"},
        {"name": "(Galaxy) DeepSeek module tests",
         "test_type": "module",
         "timeout": 150,
         "cmd": "uv pip install -r models/demos/deepseek_v3/reference/deepseek/requirements.txt && python3 models/demos/deepseek_v3/scripts/validate_weight_cache.py --root \\\"$DEEPSEEK_V3_CACHE/tests_cache\\\" || true && pytest models/demos/deepseek_v3/tests --ignore=models/demos/deepseek_v3/tests/unit --timeout 600 --durations=0"},
        {"name": "(Galaxy) DeepSeek long-seq module tests",
         "test_type": "long-seq-module",
         "timeout": 115,
         "cmd": "uv pip install -r models/demos/deepseek_v3/reference/deepseek/requirements.txt && DEEPSEEK_MAX_SEQ_LEN_OVERRIDE=16384 pytest models/demos/deepseek_v3/tests/test_mla.py -k mode_decode --timeout 900 --durations=0 && DEEPSEEK_MAX_SEQ_LEN_OVERRIDE=12288 pytest models/demos/deepseek_v3/tests/test_decoder_block.py -k mode_decode --timeout 2400 --durations=0 && DEEPSEEK_MAX_SEQ_LEN_OVERRIDE=1024 pytest models/demos/deepseek_v3/tests/test_model.py -k mode_decode --timeout 1250 --durations=0 && DEEPSEEK_MAX_SEQ_LEN_OVERRIDE=32768 pytest models/demos/deepseek_v3/tests/test_model.py -k mode_prefill --timeout 1250 --durations=0"}
    ]'
    MATRIX_FILE="$(create_matrix_file "$MATRIX_JSON")"
fi

TASK_ID="$(get_array_task_id)"
TEST_CMD="$(get_task_field "$MATRIX_FILE" "$TASK_ID" cmd)"
TEST_NAME="$(get_task_field "$MATRIX_FILE" "$TASK_ID" name)"

log_info "Running array task ${TASK_ID}: ${TEST_NAME}"

# ---------------------------------------------------------------------------
# Container execution
# ---------------------------------------------------------------------------
export DOCKER_EXTRA_ENV="DEEPSEEK_V3_HF_MODEL=${MLPERF_BASE}/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528
DEEPSEEK_V3_CACHE=${MLPERF_BASE}/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI
MESH_DEVICE=TG
LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib"
export DOCKER_EXTRA_VOLUMES="${MLPERF_BASE}:${MLPERF_BASE}:ro"

docker_run "$DOCKER_IMAGE" "
    ${TEST_CMD}
"

log_info "Galaxy DeepSeek test '${TEST_NAME}' complete"
