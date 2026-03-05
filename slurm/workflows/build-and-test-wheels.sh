#!/usr/bin/env bash
# build-and-test-wheels.sh - Two-stage orchestrator:
#   Stage 1: Build wheels on build partition (via wheels.sh)
#   Stage 2: Test wheels on target hardware with --dependency=afterok
#   Stage 3 (optional): Publish to internal PyPI
#
# Not itself an sbatch script — run directly from submit.sh or CI.
#
# Mirrors: .github/workflows/build-and-test-wheels.yaml
#
# Environment / flags:
#   PIPELINE_ID       (required) Pipeline identifier
#   PYTHON_VERSION    Python version to build wheel for (default: 3.10)
#   TEST_PARTITION    Slurm partition for tests (default: wh-n150)
#   TEST_TIMEOUT      Time limit for test jobs (default: 02:00:00)
#   TRACY             true/false — profiler support
#   ENABLE_LTO        true/false — link-time optimization
#   PUBLISH_WHEELS    true/false — publish to internal PyPI after test (default: false)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source "${SCRIPT_DIR}/_helpers/submit_dependent.sh"
source_config env

require_env PIPELINE_ID

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TEST_PARTITION="${TEST_PARTITION:-wh-n150}"
TEST_TIMEOUT="${TEST_TIMEOUT:-02:00:00}"
TRACY="${TRACY:-true}"
ENABLE_LTO="${ENABLE_LTO:-false}"
PUBLISH_WHEELS="${PUBLISH_WHEELS:-false}"

log_info "=== Build & Test Wheels pipeline ==="
log_info "  Pipeline:       ${PIPELINE_ID}"
log_info "  Python:         ${PYTHON_VERSION}"
log_info "  Test partition: ${TEST_PARTITION}"
log_info "  Tracy:          ${TRACY}"
log_info "  Publish:        ${PUBLISH_WHEELS}"

# ---------------------------------------------------------------------------
# Stage 1: Build wheels
# ---------------------------------------------------------------------------
WHEEL_EXPORT="ALL,PIPELINE_ID=${PIPELINE_ID}"
WHEEL_EXPORT+=",PYTHON_VERSION=${PYTHON_VERSION}"
WHEEL_EXPORT+=",TRACY=${TRACY}"
WHEEL_EXPORT+=",ENABLE_LTO=${ENABLE_LTO}"
[[ -n "${DOCKER_IMAGE:-}" ]] && WHEEL_EXPORT+=",DOCKER_IMAGE=${DOCKER_IMAGE}"

WHEEL_JOB_ID="$(sbatch \
    --parsable \
    --job-name="wheels-build-${PIPELINE_ID}" \
    --partition=build \
    --time=01:30:00 \
    --cpus-per-task=16 \
    --mem=32G \
    --output="logs/wheels-build-${PIPELINE_ID}-%j.out" \
    --export="${WHEEL_EXPORT}" \
    "${SCRIPT_DIR}/wheels.sh")"

log_info "Wheel build job submitted: ${WHEEL_JOB_ID}"

# ---------------------------------------------------------------------------
# Stage 2: Test wheels on hardware
# ---------------------------------------------------------------------------
# The test runner fetches the wheel from shared storage, installs it,
# and runs end-to-end tests.

TEST_EXPORT="ALL,PIPELINE_ID=${PIPELINE_ID}"
TEST_EXPORT+=",PYTHON_VERSION=${PYTHON_VERSION}"

TEST_JOB_ID="$(sbatch \
    --parsable \
    --dependency="afterok:${WHEEL_JOB_ID}" \
    --job-name="wheels-test-${PIPELINE_ID}" \
    --partition="${TEST_PARTITION}" \
    --time="${TEST_TIMEOUT}" \
    --gres=tenstorrent:1 \
    --output="logs/wheels-test-${PIPELINE_ID}-%j.out" \
    --export="${TEST_EXPORT}" \
    "${SCRIPT_DIR}/_wheel_test_runner.sh")"

log_info "Wheel test job submitted: ${TEST_JOB_ID} (depends on build: ${WHEEL_JOB_ID})"

# ---------------------------------------------------------------------------
# Stage 3 (optional): Publish wheels
# ---------------------------------------------------------------------------
PUBLISH_JOB_ID=""
if [[ "${PUBLISH_WHEELS}" == "true" ]]; then
    PUBLISH_EXPORT="ALL,PIPELINE_ID=${PIPELINE_ID}"
    PUBLISH_EXPORT+=",PYTHON_VERSION=${PYTHON_VERSION}"

    PUBLISH_JOB_ID="$(sbatch \
        --parsable \
        --dependency="afterok:${TEST_JOB_ID}" \
        --job-name="wheels-publish-${PIPELINE_ID}" \
        --partition=build \
        --time=00:30:00 \
        --output="logs/wheels-publish-${PIPELINE_ID}-%j.out" \
        --export="${PUBLISH_EXPORT}" \
        "${SCRIPT_DIR}/_wheel_publish.sh")"

    log_info "Wheel publish job submitted: ${PUBLISH_JOB_ID} (depends on test: ${TEST_JOB_ID})"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log_info "=== Pipeline submitted ==="
log_info "  Wheel Build:   ${WHEEL_JOB_ID}"
log_info "  Wheel Test:    ${TEST_JOB_ID}"
[[ -n "${PUBLISH_JOB_ID}" ]] && log_info "  Wheel Publish: ${PUBLISH_JOB_ID}"

TAIL_JOB="${PUBLISH_JOB_ID:-${TEST_JOB_ID}}"
echo "${TAIL_JOB}"
