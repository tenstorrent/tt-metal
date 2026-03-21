#!/usr/bin/env bash
# Orchestrator: PR-triggered post-commit pipeline.
# Equivalent to .github/workflows/all-post-commit-pr-workflow.yaml.
#
# Runs on a login node (no #SBATCH directives).
#
# In the GHA world this simply calls all-post-commit-workflows for fork PRs.
# Here we add optional change-file filtering so PRs that only touch docs or CI
# configs can skip heavy test suites, and we always run static checks in
# parallel with the build.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/workflows/_helpers/submit_dependent.sh"

parse_common_args "$@"

log_info "=== all-post-commit-pr-workflow orchestrator ==="
log_info "Pipeline: ${PIPELINE_ID}"
log_info "Git ref:  ${GIT_REF} (${GIT_SHORT_SHA})"

WORKFLOW_DIR="${SCRIPT_DIR}/workflows"

# ---------------------------------------------------------------------------
# Change detection — detect which files have changed vs. the merge base.
# Set SKIP_CHANGE_DETECTION=1 to force all suites to run.
# ---------------------------------------------------------------------------
RUN_TTNN=true
RUN_OPS=true
RUN_MODELS=true
RUN_TT_TRAIN=true
RUN_PROFILER=true
RUN_T3K_FAST=true

if [[ "${SKIP_CHANGE_DETECTION:-0}" != "1" ]]; then
    MERGE_BASE="${MERGE_BASE:-$(git merge-base HEAD origin/main 2>/dev/null || echo "")}"

    if [[ -n "$MERGE_BASE" ]]; then
        CHANGED_FILES="$(git diff --name-only "$MERGE_BASE"..HEAD 2>/dev/null || echo "")"

        has_changes_in() {
            local pattern="$1"
            echo "$CHANGED_FILES" | grep -qE "$pattern"
        }

        # If only docs/CI config changed, skip hardware tests
        if ! has_changes_in '^(tt_metal|tt_eager|ttnn|models|tests|tt_train)/'; then
            log_info "No source code changes detected — skipping heavy test suites"
            RUN_TTNN=false
            RUN_OPS=false
            RUN_MODELS=false
            RUN_TT_TRAIN=false
            RUN_PROFILER=false
            RUN_T3K_FAST=false
        fi

        # Selective re-enablement based on changed paths
        if has_changes_in '^ttnn/'; then
            RUN_TTNN=true
            RUN_OPS=true
        fi
        if has_changes_in '^models/'; then
            RUN_MODELS=true
        fi
        if has_changes_in '^tt_train/'; then
            RUN_TT_TRAIN=true
        fi
        if has_changes_in '^tt_metal/.*profiler'; then
            RUN_PROFILER=true
        fi
    else
        log_warn "Could not determine merge base — running all suites"
    fi
fi

# Common env exports for child jobs
COMMON_EXPORTS="ALL,PIPELINE_ID=${PIPELINE_ID}"

# ---------------------------------------------------------------------------
# Static checks (independent, runs in parallel with everything)
# ---------------------------------------------------------------------------
STATIC_JOB="$(submit_after "" \
    "${WORKFLOW_DIR}/all-static-checks.sh" \
    --partition=build --time=00:30:00 \
    --export="${COMMON_EXPORTS}")"
log_info "all-static-checks:        ${STATIC_JOB}"

# ---------------------------------------------------------------------------
# Build artifact
# ---------------------------------------------------------------------------
BUILD_JOB="$(submit_after "" \
    "${WORKFLOW_DIR}/build-artifact.sh" \
    --partition=build --time=02:00:00 \
    --export="${COMMON_EXPORTS},BUILD_TYPE=Release,BUILD_WHEEL=1")"
log_info "Build job:                 ${BUILD_JOB}"

# ---------------------------------------------------------------------------
# Fan-out test suites (depend on build)
# ---------------------------------------------------------------------------
declare -a ALL_JOBS=("${STATIC_JOB}" "${BUILD_JOB}")

if [[ "$RUN_TTNN" == "true" ]]; then
    TTNN_JOB="$(submit_after "${BUILD_JOB}" \
        "${WORKFLOW_DIR}/ttnn-post-commit.sh" \
        --partition=wh-n150 --time=02:00:00 \
        --export="${COMMON_EXPORTS},DEPENDENCY_JOBID=${BUILD_JOB},ENABLED_SKUS=wh_n300_civ2")"
    ALL_JOBS+=("${TTNN_JOB}")
    log_info "ttnn-post-commit:          ${TTNN_JOB}"
fi

if [[ "$RUN_OPS" == "true" ]]; then
    OPS_JOB="$(submit_after "${BUILD_JOB}" \
        "${WORKFLOW_DIR}/ops-post-commit.sh" \
        --partition=wh-n150 --time=02:00:00 \
        --export="${COMMON_EXPORTS}")"
    ALL_JOBS+=("${OPS_JOB}")
    log_info "ops-post-commit:           ${OPS_JOB}"
fi

if [[ "$RUN_MODELS" == "true" ]]; then
    MODELS_JOB="$(submit_after "${BUILD_JOB}" \
        "${WORKFLOW_DIR}/models-post-commit.sh" \
        --partition=wh-n150 --time=04:00:00 \
        --export="${COMMON_EXPORTS}")"
    ALL_JOBS+=("${MODELS_JOB}")
    log_info "models-post-commit:        ${MODELS_JOB}"
fi

if [[ "$RUN_TT_TRAIN" == "true" ]]; then
    TT_TRAIN_JOB="$(submit_after "${BUILD_JOB}" \
        "${WORKFLOW_DIR}/tt-train-post-commit.sh" \
        --partition=wh-n150 --time=02:00:00 \
        --export="${COMMON_EXPORTS}")"
    ALL_JOBS+=("${TT_TRAIN_JOB}")
    log_info "tt-train-post-commit:      ${TT_TRAIN_JOB}"
fi

if [[ "$RUN_PROFILER" == "true" ]]; then
    PROFILER_JOB="$(submit_after "${BUILD_JOB}" \
        "${WORKFLOW_DIR}/run-profiler-regression.sh" \
        --partition=wh-n150 --time=02:00:00 \
        --export="${COMMON_EXPORTS}")"
    ALL_JOBS+=("${PROFILER_JOB}")
    log_info "run-profiler-regression:   ${PROFILER_JOB}"
fi

if [[ "$RUN_T3K_FAST" == "true" ]]; then
    T3K_JOB="$(submit_after "${BUILD_JOB}" \
        "${WORKFLOW_DIR}/t3000-fast-tests.sh" \
        --partition=wh-t3k --time=04:00:00 \
        --export="${COMMON_EXPORTS}")"
    ALL_JOBS+=("${T3K_JOB}")
    log_info "t3000-fast-tests:          ${T3K_JOB}"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log_info "=== All PR post-commit jobs submitted (${#ALL_JOBS[@]} total) ==="
log_info "  Static checks:           ${STATIC_JOB}"
log_info "  Build:                   ${BUILD_JOB}"
[[ "$RUN_TTNN"      == "true" ]] && log_info "  TTNN:                    ${TTNN_JOB:-skipped}"
[[ "$RUN_OPS"       == "true" ]] && log_info "  Ops:                     ${OPS_JOB:-skipped}"
[[ "$RUN_MODELS"    == "true" ]] && log_info "  Models:                  ${MODELS_JOB:-skipped}"
[[ "$RUN_TT_TRAIN"  == "true" ]] && log_info "  TT-Train:                ${TT_TRAIN_JOB:-skipped}"
[[ "$RUN_PROFILER"  == "true" ]] && log_info "  Profiler:                ${PROFILER_JOB:-skipped}"
[[ "$RUN_T3K_FAST"  == "true" ]] && log_info "  T3000 fast:              ${T3K_JOB:-skipped}"
