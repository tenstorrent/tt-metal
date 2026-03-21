#!/usr/bin/env bash
# Orchestrator: submits build-artifact, then fans out to all post-commit test
# suites.  Equivalent to .github/workflows/all-post-commit-workflows.yaml.
#
# Runs on a login node (no #SBATCH directives).
# Each test suite is individually toggleable via RUN_<SUITE>=true|false env vars.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/workflows/_helpers/submit_dependent.sh"

parse_common_args "$@"

log_info "=== all-post-commit-workflows orchestrator ==="
log_info "Pipeline: ${PIPELINE_ID}"
log_info "Git ref:  ${GIT_REF} (${GIT_SHORT_SHA})"

# ---------------------------------------------------------------------------
# Configuration — which test suites to run (all default to true)
# ---------------------------------------------------------------------------
RUN_TTNN="${RUN_TTNN:-true}"
RUN_MODELS="${RUN_MODELS:-true}"
RUN_TT_TRAIN="${RUN_TT_TRAIN:-true}"
RUN_PROFILER="${RUN_PROFILER:-true}"
RUN_T3K_FAST="${RUN_T3K_FAST:-true}"
RUN_OPS="${RUN_OPS:-true}"

BUILD_TYPE="${BUILD_TYPE:-Release}"
PLATFORM="${PLATFORM:-Ubuntu 22.04}"
ENABLE_WATCHER="${ENABLE_WATCHER:-0}"
ENABLE_LIGHTWEIGHT_ASSERTS="${ENABLE_LIGHTWEIGHT_ASSERTS:-0}"
ENABLE_LLK_ASSERTS="${ENABLE_LLK_ASSERTS:-0}"

WORKFLOW_DIR="${SCRIPT_DIR}/workflows"

# ---------------------------------------------------------------------------
# Common sbatch env exports forwarded to every child job
# ---------------------------------------------------------------------------
COMMON_EXPORTS="ALL,PIPELINE_ID=${PIPELINE_ID}"
COMMON_EXPORTS+=",BUILD_TYPE=${BUILD_TYPE}"
COMMON_EXPORTS+=",ENABLE_WATCHER=${ENABLE_WATCHER}"
COMMON_EXPORTS+=",ENABLE_LIGHTWEIGHT_ASSERTS=${ENABLE_LIGHTWEIGHT_ASSERTS}"
COMMON_EXPORTS+=",ENABLE_LLK_ASSERTS=${ENABLE_LLK_ASSERTS}"

# ---------------------------------------------------------------------------
# Stage 1: Build artifact
# ---------------------------------------------------------------------------
BUILD_JOB="$(submit_after "" \
    "${WORKFLOW_DIR}/build-artifact.sh" \
    --partition=build --time=02:00:00 \
    --export="${COMMON_EXPORTS}")"
log_info "Build job: ${BUILD_JOB}"

# ---------------------------------------------------------------------------
# Stage 2: Fan-out test suites (all depend on successful build)
# ---------------------------------------------------------------------------
declare -a ALL_JOBS=("${BUILD_JOB}")

if [[ "$RUN_TTNN" == "true" ]]; then
    TTNN_JOB="$(submit_after "${BUILD_JOB}" \
        "${WORKFLOW_DIR}/ttnn-post-commit.sh" \
        --partition=wh-n150 --time=02:00:00 \
        --export="${COMMON_EXPORTS},DEPENDENCY_JOBID=${BUILD_JOB},ENABLED_SKUS=wh_n300_civ2")"
    ALL_JOBS+=("${TTNN_JOB}")
    log_info "ttnn-post-commit:       ${TTNN_JOB}"
fi

if [[ "$RUN_OPS" == "true" ]]; then
    OPS_JOB="$(submit_after "${BUILD_JOB}" \
        "${WORKFLOW_DIR}/ops-post-commit.sh" \
        --partition=wh-n150 --time=02:00:00 \
        --export="${COMMON_EXPORTS}")"
    ALL_JOBS+=("${OPS_JOB}")
    log_info "ops-post-commit:        ${OPS_JOB}"
fi

if [[ "$RUN_MODELS" == "true" ]]; then
    MODELS_JOB="$(submit_after "${BUILD_JOB}" \
        "${WORKFLOW_DIR}/models-post-commit.sh" \
        --partition=wh-n150 --time=04:00:00 \
        --export="${COMMON_EXPORTS}")"
    ALL_JOBS+=("${MODELS_JOB}")
    log_info "models-post-commit:     ${MODELS_JOB}"
fi

if [[ "$RUN_TT_TRAIN" == "true" ]]; then
    TT_TRAIN_JOB="$(submit_after "${BUILD_JOB}" \
        "${WORKFLOW_DIR}/tt-train-post-commit.sh" \
        --partition=wh-n150 --time=02:00:00 \
        --export="${COMMON_EXPORTS}")"
    ALL_JOBS+=("${TT_TRAIN_JOB}")
    log_info "tt-train-post-commit:   ${TT_TRAIN_JOB}"
fi

if [[ "$RUN_PROFILER" == "true" ]]; then
    PROFILER_JOB="$(submit_after "${BUILD_JOB}" \
        "${WORKFLOW_DIR}/run-profiler-regression.sh" \
        --partition=wh-n150 --time=02:00:00 \
        --export="${COMMON_EXPORTS}")"
    ALL_JOBS+=("${PROFILER_JOB}")
    log_info "run-profiler-regression: ${PROFILER_JOB}"
fi

if [[ "$RUN_T3K_FAST" == "true" ]]; then
    T3K_JOB="$(submit_after "${BUILD_JOB}" \
        "${WORKFLOW_DIR}/t3000-fast-tests.sh" \
        --partition=wh-t3k --time=04:00:00 \
        --export="${COMMON_EXPORTS}")"
    ALL_JOBS+=("${T3K_JOB}")
    log_info "t3000-fast-tests:       ${T3K_JOB}"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log_info "=== All post-commit jobs submitted (${#ALL_JOBS[@]} total) ==="
log_info "  Build:                ${BUILD_JOB}"
[[ "$RUN_TTNN"      == "true" ]] && log_info "  TTNN:                 ${TTNN_JOB:-skipped}"
[[ "$RUN_OPS"       == "true" ]] && log_info "  Ops:                  ${OPS_JOB:-skipped}"
[[ "$RUN_MODELS"    == "true" ]] && log_info "  Models:               ${MODELS_JOB:-skipped}"
[[ "$RUN_TT_TRAIN"  == "true" ]] && log_info "  TT-Train:             ${TT_TRAIN_JOB:-skipped}"
[[ "$RUN_PROFILER"  == "true" ]] && log_info "  Profiler:             ${PROFILER_JOB:-skipped}"
[[ "$RUN_T3K_FAST"  == "true" ]] && log_info "  T3000 fast:           ${T3K_JOB:-skipped}"
