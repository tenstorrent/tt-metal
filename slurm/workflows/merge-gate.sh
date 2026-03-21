#!/usr/bin/env bash
# Orchestrator: merge-gate pipeline — the minimum bar a PR must pass before
# it can be merged to main.  Equivalent to .github/workflows/merge-gate.yaml.
#
# Runs on a login node (no #SBATCH directives).
#
# Dependency graph:
#   (independent)  static-checks
#   (independent)  docker-images
#   docker-images -> code-analysis
#   docker-images -> build (Release)
#   docker-images -> build-tsan
#   docker-images -> build-asan
#   docker-images -> build-sweeps
#   build-tsan    -> smoke-metalium, smoke-ttnn
#   build-asan    -> basic-metalium, basic-ttnn
#   build         -> ttnn-merge-gate, tt-cnn, fabric, fabric-cpu-only, ttsim, triage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/workflows/_helpers/submit_dependent.sh"

parse_common_args "$@"

log_info "=== merge-gate orchestrator ==="
log_info "Pipeline: ${PIPELINE_ID}"
log_info "Git ref:  ${GIT_REF} (${GIT_SHORT_SHA})"

WORKFLOW_DIR="${SCRIPT_DIR}/workflows"

# ---------------------------------------------------------------------------
# Change detection (mirrors find-changed-files GHA action)
# Callers may pre-set these to skip detection.
# ---------------------------------------------------------------------------
ANY_CODE_CHANGED="${ANY_CODE_CHANGED:-true}"
CMAKE_CHANGED="${CMAKE_CHANGED:-true}"
TT_METALIUM_CHANGED="${TT_METALIUM_CHANGED:-true}"
TT_NN_CHANGED="${TT_NN_CHANGED:-true}"
TESTS_CHANGED="${TESTS_CHANGED:-true}"
BUILD_WORKFLOWS_CHANGED="${BUILD_WORKFLOWS_CHANGED:-true}"
IS_MAIN="${IS_MAIN:-false}"

if [[ "$GIT_REF" == "main" ]]; then
    IS_MAIN=true
fi

should_run_tests() {
    [[ "$IS_MAIN" == "true" ]] && return 0
    [[ "$CMAKE_CHANGED" == "true" ]] && return 0
    [[ "$ANY_CODE_CHANGED" == "true" ]] && return 0
    return 1
}

should_run_metalium_tests() {
    [[ "$IS_MAIN" == "true" ]] && return 0
    [[ "$CMAKE_CHANGED" == "true" ]] && return 0
    [[ "$TT_METALIUM_CHANGED" == "true" ]] && return 0
    [[ "$TESTS_CHANGED" == "true" ]] && return 0
    return 1
}

should_run_ttnn_tests() {
    should_run_metalium_tests && return 0
    [[ "$TT_NN_CHANGED" == "true" ]] && return 0
    return 1
}

# Common env exports for child jobs
COMMON_EXPORTS="ALL,PIPELINE_ID=${PIPELINE_ID}"

# ---------------------------------------------------------------------------
# Stage 0: Static checks + Docker images (independent, no build dependency)
# ---------------------------------------------------------------------------
STATIC_JOB="$(submit_after "" \
    "${WORKFLOW_DIR}/all-static-checks.sh" \
    --partition=build --time=00:30:00 \
    --export="${COMMON_EXPORTS}")"
log_info "all-static-checks:        ${STATIC_JOB}"

DOCKER_JOB="$(submit_after "" \
    "${WORKFLOW_DIR}/build-all-docker-images.sh" \
    --partition=build --time=02:00:00 \
    --export="${COMMON_EXPORTS}")"
log_info "build-all-docker-images:   ${DOCKER_JOB}"

# Code analysis depends on docker images
CODE_ANALYSIS_JOB="$(submit_after "${DOCKER_JOB}" \
    "${WORKFLOW_DIR}/code-analysis.sh" \
    --partition=build --time=01:00:00 \
    --export="${COMMON_EXPORTS}")"
log_info "code-analysis:             ${CODE_ANALYSIS_JOB}"

# ---------------------------------------------------------------------------
# Stage 1: Build variants (depend on docker images)
# ---------------------------------------------------------------------------
declare -a ALL_JOBS=("${STATIC_JOB}" "${DOCKER_JOB}" "${CODE_ANALYSIS_JOB}")

if should_run_tests; then
    # Release build
    BUILD_JOB="$(submit_after "${DOCKER_JOB}" \
        "${WORKFLOW_DIR}/build-artifact.sh" \
        --partition=build --time=02:00:00 \
        --export="${COMMON_EXPORTS},BUILD_TYPE=Release,BUILD_WHEEL=1")"
    ALL_JOBS+=("${BUILD_JOB}")
    log_info "build (Release):           ${BUILD_JOB}"

    # TSan build
    BUILD_TSAN_JOB="$(submit_after "${DOCKER_JOB}" \
        "${WORKFLOW_DIR}/build-artifact.sh" \
        --partition=build --time=02:00:00 \
        --job-name=build-artifact-tsan \
        --export="${COMMON_EXPORTS},BUILD_TYPE=TSan,PLATFORM=Ubuntu 24.04")"
    ALL_JOBS+=("${BUILD_TSAN_JOB}")
    log_info "build (TSan):              ${BUILD_TSAN_JOB}"

    # ASan build
    BUILD_ASAN_JOB="$(submit_after "${DOCKER_JOB}" \
        "${WORKFLOW_DIR}/build-artifact.sh" \
        --partition=build --time=02:00:00 \
        --job-name=build-artifact-asan \
        --export="${COMMON_EXPORTS},BUILD_TYPE=ASan")"
    ALL_JOBS+=("${BUILD_ASAN_JOB}")
    log_info "build (ASan):              ${BUILD_ASAN_JOB}"

    # Sweeps build
    BUILD_SWEEPS_JOB="$(submit_after "${DOCKER_JOB}" \
        "${WORKFLOW_DIR}/build-wrapper.sh" \
        --partition=build --time=02:00:00 \
        --export="${COMMON_EXPORTS}")"
    ALL_JOBS+=("${BUILD_SWEEPS_JOB}")
    log_info "build-sweeps:              ${BUILD_SWEEPS_JOB}"
fi

# ---------------------------------------------------------------------------
# Stage 2a: TSan-linked tests (smoke)
# ---------------------------------------------------------------------------
if should_run_metalium_tests && [[ -n "${BUILD_TSAN_JOB:-}" ]]; then
    for product in tt-metalium tt-nn; do
        SMOKE_JOB="$(submit_after "${BUILD_TSAN_JOB}" \
            "${WORKFLOW_DIR}/smoke.sh" \
            --partition=wh-n150 --time=00:30:00 \
            --job-name="smoke-${product}" \
            --export="${COMMON_EXPORTS},PRODUCT=${product},PER_TEST_TIMEOUT=11")"
        ALL_JOBS+=("${SMOKE_JOB}")
        log_info "smoke (${product}):         ${SMOKE_JOB}"
    done
fi

# ---------------------------------------------------------------------------
# Stage 2b: ASan-linked tests (basic)
# ---------------------------------------------------------------------------
if should_run_metalium_tests && [[ -n "${BUILD_ASAN_JOB:-}" ]]; then
    for product in tt-metalium tt-nn; do
        BASIC_JOB="$(submit_after "${BUILD_ASAN_JOB}" \
            "${WORKFLOW_DIR}/basic.sh" \
            --partition=wh-n150 --time=01:00:00 \
            --job-name="basic-${product}" \
            --export="${COMMON_EXPORTS},PRODUCT=${product},PER_TEST_TIMEOUT=11")"
        ALL_JOBS+=("${BASIC_JOB}")
        log_info "basic (${product}):         ${BASIC_JOB}"
    done
fi

# ---------------------------------------------------------------------------
# Stage 2c: Release-linked tests
# ---------------------------------------------------------------------------
if should_run_ttnn_tests && [[ -n "${BUILD_JOB:-}" ]]; then
    TTNN_MG_JOB="$(submit_after "${BUILD_JOB}" \
        "${WORKFLOW_DIR}/ttnn-post-commit.sh" \
        --partition=wh-n150 --time=02:00:00 \
        --export="${COMMON_EXPORTS},DEPENDENCY_JOBID=${BUILD_JOB},ENABLED_SKUS=wh_n300_civ2,MERGE_GATE_CALL=true")"
    ALL_JOBS+=("${TTNN_MG_JOB}")
    log_info "ttnn-merge-gate:           ${TTNN_MG_JOB}"

    CNN_JOB="$(submit_after "${BUILD_JOB}" \
        "${WORKFLOW_DIR}/tt-cnn-post-commit.sh" \
        --partition=wh-n150 --time=02:00:00 \
        --export="${COMMON_EXPORTS}")"
    ALL_JOBS+=("${CNN_JOB}")
    log_info "tt-cnn-post-commit:        ${CNN_JOB}"

    FABRIC_JOB="$(submit_after "${BUILD_JOB}" \
        "${WORKFLOW_DIR}/fabric-build-and-unit-tests.sh" \
        --partition=wh-n150 --time=02:00:00 \
        --export="${COMMON_EXPORTS}")"
    ALL_JOBS+=("${FABRIC_JOB}")
    log_info "fabric-unit-tests:         ${FABRIC_JOB}"

    FABRIC_CPU_JOB="$(submit_after "${BUILD_JOB}" \
        "${WORKFLOW_DIR}/fabric-cpu-only-tests.sh" \
        --partition=build --time=01:00:00 \
        --export="${COMMON_EXPORTS}")"
    ALL_JOBS+=("${FABRIC_CPU_JOB}")
    log_info "fabric-cpu-only:           ${FABRIC_CPU_JOB}"

    TTSIM_JOB="$(submit_after "${BUILD_JOB}" \
        "${WORKFLOW_DIR}/ttsim.sh" \
        --partition=build --time=01:00:00 \
        --export="${COMMON_EXPORTS}")"
    ALL_JOBS+=("${TTSIM_JOB}")
    log_info "ttsim:                     ${TTSIM_JOB}"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log_info "=== All merge-gate jobs submitted (${#ALL_JOBS[@]} total) ==="
printf '  %s\n' "${ALL_JOBS[@]}" | while IFS= read -r jid; do
    log_info "  JOBID: ${jid}"
done
