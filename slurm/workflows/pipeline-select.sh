#!/usr/bin/env bash
# pipeline-select.sh - Orchestrator: conditionally select and submit pipelines
# based on changed files in the commit.
#
# Analyzes git diff to determine which test pipelines are needed,
# then submits only the relevant ones.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source "${SCRIPT_DIR}/_helpers/submit_dependent.sh"

require_env PIPELINE_ID

log_info "=== Pipeline selector ==="
log_info "  Pipeline: ${PIPELINE_ID}"

cd "${REPO_ROOT}"

CHANGED_FILES="$(git diff --name-only HEAD~1 HEAD 2>/dev/null || echo '')"

RUN_CPP=0
RUN_TTNN=0
RUN_MODELS=0
RUN_INFRA=0
RUN_PROFILER=0
RUN_FABRIC=0

while IFS= read -r file; do
    [[ -z "${file}" ]] && continue
    case "${file}" in
        tt_metal/*)           RUN_CPP=1 ;;
        ttnn/*)               RUN_TTNN=1 ;;
        models/*)             RUN_MODELS=1 ;;
        tests/*)              RUN_CPP=1; RUN_TTNN=1 ;;
        .github/*|infra/*)    RUN_INFRA=1 ;;
        *profiler*|*tracy*)   RUN_PROFILER=1 ;;
        tt_fabric/*)          RUN_FABRIC=1 ;;
        CMakeLists.txt|cmake/*) RUN_CPP=1 ;;
    esac
done <<< "${CHANGED_FILES}"

if [[ "${FORCE_ALL:-0}" == "1" ]]; then
    RUN_CPP=1 RUN_TTNN=1 RUN_MODELS=1 RUN_INFRA=1 RUN_PROFILER=1 RUN_FABRIC=1
fi

# Always build first
BUILD_JOB=$(submit_after "" "${SCRIPT_DIR}/build-artifact.sh" \
    --partition=build)
log_info "Build: ${BUILD_JOB}"

SUBMITTED=""

if [[ ${RUN_CPP} -eq 1 ]]; then
    JOB=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/cpp-post-commit.sh" \
        --partition=wh-n150 2>/dev/null) && SUBMITTED="${SUBMITTED} cpp:${JOB}" || true
fi

if [[ ${RUN_TTNN} -eq 1 ]]; then
    JOB=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/fast-dispatch-frequent-tests.sh" \
        --partition=wh-n150 2>/dev/null) && SUBMITTED="${SUBMITTED} ttnn:${JOB}" || true
fi

if [[ ${RUN_MODELS} -eq 1 ]]; then
    JOB=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/fast-dispatch-full-regressions.sh" \
        --partition=wh-n150 2>/dev/null) && SUBMITTED="${SUBMITTED} models:${JOB}" || true
fi

if [[ ${RUN_INFRA} -eq 1 ]]; then
    JOB=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/unit-tests-infra.sh" \
        --partition=build) && SUBMITTED="${SUBMITTED} infra:${JOB}" || true
fi

if [[ ${RUN_PROFILER} -eq 1 ]]; then
    JOB=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/profiler-tests.sh" \
        --partition=wh-n150 2>/dev/null) && SUBMITTED="${SUBMITTED} profiler:${JOB}" || true
fi

if [[ ${RUN_FABRIC} -eq 1 ]]; then
    JOB=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/fabric-build-and-unit-tests.sh" \
        --partition=wh-n150 2>/dev/null) && SUBMITTED="${SUBMITTED} fabric:${JOB}" || true
fi

log_info "=== Pipeline selector complete ==="
log_info "  Submitted:${SUBMITTED:-none}"
