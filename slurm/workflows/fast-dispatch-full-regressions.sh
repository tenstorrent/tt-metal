#!/usr/bin/env bash
# Orchestrator: submits build then fast-dispatch-full-regressions-impl as array jobs
#
# GHA source: .github/workflows/fast-dispatch-full-regressions-and-models-impl.yaml
# Matrix: stable models x {N150,N300} + unstable models x {N150,N300} + CIv2 models
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/workflows/_helpers/submit_dependent.sh"

parse_common_args "$@"

log_info "=== fast-dispatch-full-regressions orchestrator ==="
log_info "Pipeline: ${PIPELINE_ID}"

MODELS_TO_RUN="${MODELS_TO_RUN:-all}"
ENABLE_OPS_RECORDING="${ENABLE_OPS_RECORDING:-false}"

BUILD_JOB=$(submit_after "" \
    "${SCRIPT_DIR}/workflows/build-artifact.sh" \
    --partition=build --time=02:00:00)
log_info "Build job: ${BUILD_JOB}"

# The impl script handles the model catalog internally via TASK_ID.
# Total stable models: 24 x 2 cards = 48 tasks
# Total unstable models: up to 6 x 2 cards = 12 tasks
# Total CIv2 models: 2 x 2 cards = 4 tasks
# We submit separate array jobs for each category.

STABLE_COUNT="${FD_FULL_STABLE_COUNT:-24}"
CARDS=2  # N150, N300
STABLE_TASKS=$(( STABLE_COUNT * CARDS ))

STABLE_JOB=$(sbatch \
    --parsable \
    --dependency="afterok:${BUILD_JOB}" \
    --array="0-$((STABLE_TASKS - 1))" \
    --partition=wh-n300 \
    --time=04:00:00 \
    --job-name=fd-full-regressions-stable \
    --output="${LOG_DIR}/%x-%j-%a.out" \
    --export="ALL,PIPELINE_ID=${PIPELINE_ID},MODEL_SET=stable,MODELS_TO_RUN=${MODELS_TO_RUN},ENABLE_OPS_RECORDING=${ENABLE_OPS_RECORDING}" \
    "${SCRIPT_DIR}/workflows/fast-dispatch-full-regressions-impl.sh" 2>&1 | awk '/Submitted batch job/{print $NF}')
: "${STABLE_JOB:=$(echo "$STABLE_JOB" | tail -1)}"
log_info "Submitted stable regressions array ${STABLE_JOB} (${STABLE_TASKS} tasks)"

UNSTABLE_COUNT="${FD_FULL_UNSTABLE_COUNT:-1}"
UNSTABLE_TASKS=$(( UNSTABLE_COUNT * CARDS ))

if (( UNSTABLE_TASKS > 0 )); then
    UNSTABLE_JOB=$(sbatch \
        --parsable \
        --dependency="afterok:${BUILD_JOB}" \
        --array="0-$((UNSTABLE_TASKS - 1))" \
        --partition=wh-n300 \
        --time=04:00:00 \
        --job-name=fd-full-regressions-unstable \
        --output="${LOG_DIR}/%x-%j-%a.out" \
        --export="ALL,PIPELINE_ID=${PIPELINE_ID},MODEL_SET=unstable,MODELS_TO_RUN=${MODELS_TO_RUN}" \
        "${SCRIPT_DIR}/workflows/fast-dispatch-full-regressions-impl.sh" 2>&1 | awk '/Submitted batch job/{print $NF}')
    : "${UNSTABLE_JOB:=$(echo "$UNSTABLE_JOB" | tail -1)}"
    log_info "Submitted unstable regressions array ${UNSTABLE_JOB} (${UNSTABLE_TASKS} tasks)"
fi

echo "${STABLE_JOB}"
