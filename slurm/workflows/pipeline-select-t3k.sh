#!/usr/bin/env bash
# pipeline-select-t3k.sh - Orchestrator for T3K pipeline selection
#
# Selects and submits T3K-specific test suites based on changed files
# and configuration.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source "${SCRIPT_DIR}/_helpers/submit_dependent.sh"

require_env PIPELINE_ID

log_info "=== T3K pipeline selector ==="
log_info "  Pipeline: ${PIPELINE_ID}"

cd "${REPO_ROOT}"

CHANGED_FILES="$(git diff --name-only HEAD~1 HEAD 2>/dev/null || echo '')"

RUN_UNIT=0
RUN_DEMO=0
RUN_PERF=0
RUN_FREQ=0

while IFS= read -r file; do
    [[ -z "${file}" ]] && continue
    case "${file}" in
        tt_metal/*|ttnn/*)      RUN_UNIT=1; RUN_FREQ=1 ;;
        models/*)                RUN_DEMO=1; RUN_PERF=1 ;;
        tests/*t3k*|tests/*t3000*) RUN_UNIT=1; RUN_DEMO=1 ;;
        tt_fabric/*)             RUN_UNIT=1 ;;
    esac
done <<< "${CHANGED_FILES}"

if [[ "${FORCE_ALL:-0}" == "1" ]]; then
    RUN_UNIT=1 RUN_DEMO=1 RUN_PERF=1 RUN_FREQ=1
fi

BUILD_JOB=$(submit_after "" "${SCRIPT_DIR}/build-artifact.sh" \
    --partition=build)
log_info "Build: ${BUILD_JOB}"

SUBMITTED=""

if [[ ${RUN_UNIT} -eq 1 ]]; then
    JOB=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/t3000-unit-tests.sh" \
        --partition=wh-t3k 2>/dev/null) && SUBMITTED="${SUBMITTED} unit:${JOB}" || true
fi

if [[ ${RUN_DEMO} -eq 1 ]]; then
    JOB=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/t3000-demo-tests.sh" \
        --partition=wh-t3k 2>/dev/null) && SUBMITTED="${SUBMITTED} demo:${JOB}" || true
fi

if [[ ${RUN_PERF} -eq 1 ]]; then
    JOB=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/t3000-perf-tests.sh" \
        --partition=wh-t3k 2>/dev/null) && SUBMITTED="${SUBMITTED} perf:${JOB}" || true
fi

if [[ ${RUN_FREQ} -eq 1 ]]; then
    JOB=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/t3000-fast-tests.sh" \
        --partition=wh-t3k 2>/dev/null) && SUBMITTED="${SUBMITTED} frequent:${JOB}" || true
fi

log_info "=== T3K pipeline selector complete ==="
log_info "  Submitted:${SUBMITTED:- none}"
