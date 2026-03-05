#!/usr/bin/env bash
# pipeline-select-galaxy.sh - Orchestrator for Galaxy pipeline selection
#
# Selects and submits Galaxy-specific test suites based on changed files
# and configuration.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source "${SCRIPT_DIR}/_helpers/submit_dependent.sh"

require_env PIPELINE_ID

log_info "=== Galaxy pipeline selector ==="
log_info "  Pipeline: ${PIPELINE_ID}"

cd "${REPO_ROOT}"

CHANGED_FILES="$(git diff --name-only HEAD~1 HEAD 2>/dev/null || echo '')"

RUN_UNIT=0
RUN_DEMO=0
RUN_PERF=0
RUN_E2E=0
RUN_INTEGRATION=0

while IFS= read -r file; do
    [[ -z "${file}" ]] && continue
    case "${file}" in
        tt_metal/*|ttnn/*)           RUN_UNIT=1; RUN_DEMO=1 ;;
        models/*)                     RUN_DEMO=1; RUN_PERF=1 ;;
        tests/*galaxy*|tests/*multi*) RUN_UNIT=1; RUN_E2E=1; RUN_INTEGRATION=1 ;;
        tt_fabric/*)                  RUN_INTEGRATION=1 ;;
    esac
done <<< "${CHANGED_FILES}"

if [[ "${FORCE_ALL:-0}" == "1" ]]; then
    RUN_UNIT=1 RUN_DEMO=1 RUN_PERF=1 RUN_E2E=1 RUN_INTEGRATION=1
fi

BUILD_JOB=$(submit_after "" "${SCRIPT_DIR}/build-artifact.sh" \
    --partition=build)
log_info "Build: ${BUILD_JOB}"

SUBMITTED=""

if [[ ${RUN_UNIT} -eq 1 ]]; then
    JOB=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/galaxy-unit-tests.sh" \
        --partition=wh-galaxy 2>/dev/null) && SUBMITTED="${SUBMITTED} unit:${JOB}" || true
fi

if [[ ${RUN_DEMO} -eq 1 ]]; then
    JOB=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/galaxy-demo-tests.sh" \
        --partition=wh-galaxy 2>/dev/null) && SUBMITTED="${SUBMITTED} demo:${JOB}" || true
fi

if [[ ${RUN_PERF} -eq 1 ]]; then
    JOB=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/galaxy-perf-tests.sh" \
        --partition=wh-galaxy 2>/dev/null) && SUBMITTED="${SUBMITTED} perf:${JOB}" || true
fi

if [[ ${RUN_E2E} -eq 1 ]]; then
    JOB=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/galaxy-e2e-tests.sh" \
        --partition=wh-galaxy 2>/dev/null) && SUBMITTED="${SUBMITTED} e2e:${JOB}" || true
fi

if [[ ${RUN_INTEGRATION} -eq 1 ]]; then
    JOB=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/galaxy-integration-tests.sh" \
        --partition=wh-galaxy 2>/dev/null) && SUBMITTED="${SUBMITTED} integration:${JOB}" || true
fi

log_info "=== Galaxy pipeline selector complete ==="
log_info "  Submitted:${SUBMITTED:- none}"
