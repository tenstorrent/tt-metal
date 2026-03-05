#!/usr/bin/env bash
#SBATCH --job-name=fast-dispatch-full-regressions
#SBATCH --partition=wh-n300
#SBATCH --time=04:00:00
#SBATCH --output=/weka/ci/logs/%x/%j/%a.log
#SBATCH --error=/weka/ci/logs/%x/%j/%a.err
#
# GHA source: .github/workflows/fast-dispatch-full-regressions-and-models-impl.yaml
# Runs nightly model regression tests. TASK_ID selects model + card combination.
# MODEL_SET env var selects stable vs unstable catalog.

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

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_SET="${MODEL_SET:-stable}"
ENABLE_OPS_RECORDING="${ENABLE_OPS_RECORDING:-false}"
export ARCH_NAME=wormhole_b0

# Stable model catalog (mirrors GHA create-test-matrix)
STABLE_MODELS=(
    common_models functional_unet "ttt-llama3.2-1B" "qwen25_vl-3B"
    resnet50 whisper openpdn_mnist vit sentence_bert swin_s swin_v2
    mobilenetv2 mobilenetv3 segformer vgg_unet vanilla_unet
    efficientnetb0 efficientdetd0 vovnet ssd512 yunet retinanet ufld_v2
    "ttt-mistral-7B-v0.3" stable_diffusion_xl_base stable_diffusion
)

UNSTABLE_MODELS=(
    "mamba_1" "mamba_2" "mamba_3" "mamba_4" "mamba_5" "mamba_6"
)

CARDS=(N150 N300)

if [[ "${MODEL_SET}" == "stable" ]]; then
    MODEL_COUNT=${#STABLE_MODELS[@]}
    MODEL_INDEX=$(( TASK_ID % MODEL_COUNT ))
    CARD_INDEX=$(( TASK_ID / MODEL_COUNT ))
    MODEL="${STABLE_MODELS[$MODEL_INDEX]}"

    # Per-model timeout overrides (minutes)
    TIMEOUT=30
    case "${MODEL}" in
        stable_diffusion_xl_base) TIMEOUT=75 ;;
        stable_diffusion)         TIMEOUT=45 ;;
    esac
elif [[ "${MODEL_SET}" == "unstable" ]]; then
    MODEL_COUNT=${#UNSTABLE_MODELS[@]}
    MODEL_INDEX=$(( TASK_ID % MODEL_COUNT ))
    CARD_INDEX=$(( TASK_ID / MODEL_COUNT ))
    MODEL="${UNSTABLE_MODELS[$MODEL_INDEX]}"
    TIMEOUT=60
else
    log_fatal "Unknown MODEL_SET: ${MODEL_SET}"
fi

CARD="${CARDS[$CARD_INDEX]:-N150}"
log_info "Running nightly ${MODEL_SET} model: ${MODEL} on ${CARD} (task ${TASK_ID}, timeout ${TIMEOUT}m)"

PYTEST_CMD="pytest"
if [[ "${ENABLE_OPS_RECORDING}" == "true" ]]; then
    PYTEST_CMD="python -m tracy -r -p --dump-device-data-mid-run -m pytest"
fi

# Build the test command based on model type
if [[ "${MODEL_SET}" == "unstable" ]]; then
    MAMBA_GROUP="${MODEL##*_}"
    TEST_CMD="${PYTEST_CMD} --timeout 900 tests/nightly/single_card/mamba --splits 6 --group ${MAMBA_GROUP}"
elif [[ "${MODEL}" == *"ttt"* ]]; then
    TEST_CMD="${PYTEST_CMD} tests/nightly/single_card/tt_transformers -k ${MODEL}"
elif [[ "${MODEL}" == *"qwen25_vl"* ]]; then
    TEST_CMD="${PYTEST_CMD} tests/nightly/single_card/qwen25_vl -k ${MODEL}"
else
    TEST_CMD="${PYTEST_CMD} tests/nightly/single_card/${MODEL}"
fi

EXTRA_ENV="GTEST_OUTPUT=xml:generated/test_reports/"
if [[ "${MODEL}" == "stable_diffusion_xl_base" || "${MODEL}" == "stable_diffusion" ]]; then
    EXTRA_ENV="${EXTRA_ENV}
TT_MM_THROTTLE_PERF=5"
fi
export DOCKER_EXTRA_ENV="${EXTRA_ENV}"
export DOCKER_EXTRA_VOLUMES="/mnt/MLPerf:/mnt/MLPerf:ro"

docker_run "$DOCKER_IMAGE" "${TEST_CMD}"
