#!/usr/bin/env bash
#SBATCH --job-name=t3000-perplexity-tests
#SBATCH --partition=wh-t3k
#SBATCH --constraint=pipeline-perf
#SBATCH --time=06:00:00

# T3000 perplexity/accuracy tests — array job with inline matrix of 8 model groups.
# Equivalent to .github/workflows/t3000-perplexity-tests-impl.yaml
# Runs on perf-constrained nodes; each task can take up to 300 minutes.

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
# Inline matrix (matches GHA inline strategy.matrix)
# ---------------------------------------------------------------------------
if [[ -z "${MATRIX_FILE:-}" ]]; then
    MATRIX_JSON='[
        {"name": "t3k_falcon40b_tests",           "cmd": "run_t3000_falcon40b_perplexity_tests",           "timeout": 300},
        {"name": "t3k_llama_70b_tests",            "cmd": "run_t3000_llama70b_perplexity_tests",            "timeout": 300},
        {"name": "t3k_mistral_tests",              "cmd": "run_t3000_mistral_perplexity_tests",             "timeout": 300},
        {"name": "t3k_llama3_tests_single_card",   "cmd": "run_t3000_llama3_perplexity_tests_single_card",  "timeout": 300},
        {"name": "t3k_llama3_tests_t3k",           "cmd": "run_t3000_llama3_perplexity_tests_t3000",        "timeout": 300},
        {"name": "t3k_qwen25_tests",               "cmd": "run_t3000_qwen25_perplexity_tests",             "timeout": 300},
        {"name": "t3k_qwen3_tests",                "cmd": "run_t3000_qwen3_perplexity_tests",              "timeout": 300},
        {"name": "t3k_gemma3_tests",               "cmd": "run_t3000_gemma3_accuracy_tests",               "timeout": 300}
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
export DOCKER_EXTRA_ENV="HF_HUB_CACHE=${MLPERF_BASE}/huggingface/hub
LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib"
export DOCKER_EXTRA_VOLUMES="${MLPERF_BASE}:${MLPERF_BASE}:ro"

docker_run "$DOCKER_IMAGE" "
    source \${TT_METAL_HOME}/tests/scripts/t3000/run_t3000_perplexity_tests.sh
    ${TEST_CMD}
"

log_info "T3000 perplexity test '${TEST_NAME}' complete"
