#!/usr/bin/env bash
# Orchestrator: submits build then fast-dispatch-frequent-tests-impl as array job
#
# GHA source: .github/workflows/fast-dispatch-frequent-tests-impl.yaml
# Matrix: 4 test groups (WH N300 pgm dispatch, WH N300 dispatch BW,
#         BH P150 pgm dispatch, BH P150 dispatch BW)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/workflows/_helpers/submit_dependent.sh"

parse_common_args "$@"

log_info "=== fast-dispatch-frequent-tests orchestrator ==="
log_info "Pipeline: ${PIPELINE_ID}"

NUM_GROUPS=4

BUILD_JOB=$(submit_after "" \
    "${SCRIPT_DIR}/workflows/build-artifact.sh" \
    --partition=build --time=02:00:00)
log_info "Build job: ${BUILD_JOB}"

ARRAY_SPEC="0-$((NUM_GROUPS - 1))"
TEST_JOB=$(sbatch \
    --parsable \
    --dependency="afterok:${BUILD_JOB}" \
    --array="${ARRAY_SPEC}" \
    --partition=wh-n150 \
    --time=02:00:00 \
    --job-name=fast-dispatch-frequent-tests \
    --output="${LOG_DIR}/%x-%j-%a.out" \
    --export="ALL,PIPELINE_ID=${PIPELINE_ID}" \
    "${SCRIPT_DIR}/workflows/fast-dispatch-frequent-tests-impl.sh" 2>&1 | awk '/Submitted batch job/{print $NF}')

: "${TEST_JOB:=$(echo "$TEST_JOB" | tail -1)}"

log_info "Submitted frequent tests array ${TEST_JOB} (${NUM_GROUPS} tasks, after build ${BUILD_JOB})"
echo "${TEST_JOB}"
