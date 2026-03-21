#!/usr/bin/env bash
# tt-metal-l2-nightly.sh - Orchestrator: comprehensive nightly test pipeline
#
# Submits build + a full matrix of test suites across hardware SKUs.
# Designed to be called from scrontab at 0 6 * * *.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source "${SCRIPT_DIR}/_helpers/submit_dependent.sh"

require_env PIPELINE_ID

log_info "=== L2 Nightly orchestrator ==="
log_info "  Pipeline: ${PIPELINE_ID}"
log_info "  Time:     $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

# Step 1: Build
BUILD_JOB=$(submit_after "" "${SCRIPT_DIR}/build-artifact.sh" \
    --partition=build)
log_info "Submitted build: ${BUILD_JOB}"

# Step 2: Unit tests (N150, after build)
UNIT_N150=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/unit-tests-infra.sh" \
    --partition=build)

# Step 3: Fast dispatch tests (after build)
FD_FREQUENT=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/fast-dispatch-frequent-tests.sh" \
    --partition=wh-n150)

FD_FULL=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/fast-dispatch-full-regressions.sh" \
    --partition=wh-n150)

# Step 4: Profiler tests (after build)
PROFILER=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/profiler-tests.sh" \
    --partition=wh-n150)

PROFILER_REG=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/run-profiler-regression.sh" \
    --partition=wh-n150)

# Step 5: Galaxy tests (after build)
GALAXY_UNIT=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/galaxy-unit-tests.sh" \
    --partition=wh-galaxy 2>/dev/null) || GALAXY_UNIT=""

GALAXY_DEMO=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/galaxy-demo-tests.sh" \
    --partition=wh-galaxy 2>/dev/null) || GALAXY_DEMO=""

# Step 6: T3K tests (after build)
T3K_DEMO=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/t3000-demo-tests.sh" \
    --partition=wh-t3k 2>/dev/null) || T3K_DEMO=""

# Step 7: Upstream tests (after build)
UPSTREAM=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/upstream-tests.sh" \
    --partition=wh-n150)

# Step 8: UMD unit tests (after build)
UMD=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/umd-unit-tests.sh" \
    --partition=wh-n150)

# Step 9: Notification (after everything, any exit status)
ALL_JOBS="${BUILD_JOB}"
for jid in ${UNIT_N150} ${FD_FREQUENT} ${FD_FULL} ${PROFILER} ${PROFILER_REG} \
           ${GALAXY_UNIT} ${GALAXY_DEMO} ${T3K_DEMO} ${UPSTREAM} ${UMD}; do
    [[ -n "${jid}" ]] && ALL_JOBS="${ALL_JOBS}:${jid}"
done

NOTIFY_JOB=$(submit_after_any "${ALL_JOBS##:}" "${SCRIPT_DIR}/ttnn-sweeps-slack-notify.sh" \
    --partition=build --export=ALL,PIPELINE_ID="${PIPELINE_ID}",NOTIFY_TYPE=nightly)

log_info "=== L2 Nightly pipeline submitted ==="
log_info "  Build:             ${BUILD_JOB}"
log_info "  Unit (infra):      ${UNIT_N150}"
log_info "  FD frequent:       ${FD_FREQUENT}"
log_info "  FD full:           ${FD_FULL}"
log_info "  Profiler:          ${PROFILER}"
log_info "  Profiler regr:     ${PROFILER_REG}"
log_info "  Galaxy unit:       ${GALAXY_UNIT:-skipped}"
log_info "  Galaxy demo:       ${GALAXY_DEMO:-skipped}"
log_info "  T3K demo:          ${T3K_DEMO:-skipped}"
log_info "  Upstream:          ${UPSTREAM}"
log_info "  UMD:               ${UMD}"
log_info "  Notify:            ${NOTIFY_JOB}"
