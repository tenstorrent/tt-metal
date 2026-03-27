#!/usr/bin/env bash
# profiler-ci-selector.sh - Orchestrator: select and submit profiler jobs
#
# Determines which profiler test suites to run based on changed files,
# then submits the appropriate subset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source "${SCRIPT_DIR}/_helpers/submit_dependent.sh"

require_env PIPELINE_ID

log_info "=== Profiler CI selector ==="
log_info "  Pipeline: ${PIPELINE_ID}"

cd "${REPO_ROOT}"

RUN_PROFILER=0
RUN_PROFILER_REGRESSION=0

CHANGED_FILES="$(git diff --name-only HEAD~1 HEAD 2>/dev/null || echo '')"

if echo "${CHANGED_FILES}" | grep -qE '(tt_metal/tools/profiler|ttnn/.*profil)'; then
    RUN_PROFILER=1
    RUN_PROFILER_REGRESSION=1
    log_info "Profiler source changes detected"
elif echo "${CHANGED_FILES}" | grep -qE '(tracy|profil)'; then
    RUN_PROFILER=1
    log_info "Tracy/profiler-adjacent changes detected"
fi

if [[ "${FORCE_PROFILER:-0}" == "1" ]]; then
    RUN_PROFILER=1
    RUN_PROFILER_REGRESSION=1
    log_info "FORCE_PROFILER=1, running all profiler tests"
fi

SUBMITTED_JOBS=""

if [[ ${RUN_PROFILER} -eq 1 ]]; then
    PROFILER_JOB=$(submit_after "" "${SCRIPT_DIR}/profiler-tests.sh" \
        --partition=wh-n150)
    SUBMITTED_JOBS="${SUBMITTED_JOBS} profiler-tests:${PROFILER_JOB}"
fi

if [[ ${RUN_PROFILER_REGRESSION} -eq 1 ]]; then
    REGRESSION_JOB=$(submit_after "" "${SCRIPT_DIR}/run-profiler-regression.sh" \
        --partition=wh-n150)
    SUBMITTED_JOBS="${SUBMITTED_JOBS} run-profiler-regression:${REGRESSION_JOB}"
fi

if [[ -z "${SUBMITTED_JOBS}" ]]; then
    log_info "No profiler tests needed for this changeset"
else
    log_info "Submitted profiler jobs:${SUBMITTED_JOBS}"
fi

log_info "=== Profiler CI selector complete ==="
