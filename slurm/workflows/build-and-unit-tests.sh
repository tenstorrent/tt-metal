#!/usr/bin/env bash
# build-and-unit-tests.sh - Two-stage orchestrator:
#   Stage 1: Submit build-artifact.sh on build partition
#   Stage 2: Submit unit tests on target hardware with --dependency=afterok
#
# Not itself an sbatch script — run directly from submit.sh or CI.
#
# Mirrors: .github/workflows/build-and-unit-tests.yaml
#
# Environment / flags:
#   PIPELINE_ID       (required) Pipeline identifier
#   PLATFORM          Platform string (default: "Ubuntu 22.04")
#   BUILD_TYPE        Build type (default: Release)
#   TOOLCHAIN         CMake toolchain file
#   TEST_PARTITION    Slurm partition for tests (default: wh-n150)
#   TEST_TIMEOUT      Time limit for test jobs (default: 04:00:00)
#   TRACY             true/false — profiler build
#   DISTRIBUTED       true/false — distributed build
#   DOCKER_IMAGE      Pre-built Docker image (skips docker build)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source "${SCRIPT_DIR}/_helpers/submit_dependent.sh"
source_config env

require_env PIPELINE_ID

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PLATFORM="${PLATFORM:-Ubuntu 22.04}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
TOOLCHAIN="${TOOLCHAIN:-}"
TEST_PARTITION="${TEST_PARTITION:-wh-n150}"
TEST_TIMEOUT="${TEST_TIMEOUT:-04:00:00}"
TRACY="${TRACY:-false}"
DISTRIBUTED="${DISTRIBUTED:-true}"

log_info "=== Build & Unit Tests pipeline ==="
log_info "  Pipeline:       ${PIPELINE_ID}"
log_info "  Platform:       ${PLATFORM}"
log_info "  Build type:     ${BUILD_TYPE}"
log_info "  Test partition: ${TEST_PARTITION}"

# ---------------------------------------------------------------------------
# Stage 1: Build
# ---------------------------------------------------------------------------
BUILD_EXPORT="ALL,PIPELINE_ID=${PIPELINE_ID}"
BUILD_EXPORT+=",PLATFORM=${PLATFORM}"
BUILD_EXPORT+=",BUILD_TYPE=${BUILD_TYPE}"
BUILD_EXPORT+=",TRACY=${TRACY}"
BUILD_EXPORT+=",DISTRIBUTED=${DISTRIBUTED}"
[[ -n "${TOOLCHAIN}" ]] && BUILD_EXPORT+=",TOOLCHAIN=${TOOLCHAIN}"
[[ -n "${DOCKER_IMAGE:-}" ]] && BUILD_EXPORT+=",DOCKER_IMAGE=${DOCKER_IMAGE}"

BUILD_JOB_ID="$(sbatch \
    --parsable \
    --job-name="build-${PIPELINE_ID}" \
    --partition=build \
    --time=02:00:00 \
    --cpus-per-task=16 \
    --mem=64G \
    --output="logs/build-${PIPELINE_ID}-%j.out" \
    --export="${BUILD_EXPORT}" \
    "${SCRIPT_DIR}/build-artifact.sh")"

log_info "Build job submitted: ${BUILD_JOB_ID}"

# ---------------------------------------------------------------------------
# Stage 2: Unit tests (depend on successful build)
# ---------------------------------------------------------------------------
# The unit test runner fetches the build artifact from shared Weka storage
# and runs the C++ and Python unit test suites.

TEST_EXPORT="ALL,PIPELINE_ID=${PIPELINE_ID}"
TEST_EXPORT+=",PLATFORM=${PLATFORM}"
[[ -n "${DOCKER_IMAGE:-}" ]] && TEST_EXPORT+=",DOCKER_IMAGE=${DOCKER_IMAGE}"

TEST_JOB_ID="$(sbatch \
    --parsable \
    --dependency="afterok:${BUILD_JOB_ID}" \
    --job-name="unit-tests-${PIPELINE_ID}" \
    --partition="${TEST_PARTITION}" \
    --time="${TEST_TIMEOUT}" \
    --gres=tenstorrent:1 \
    --output="logs/unit-tests-${PIPELINE_ID}-%j.out" \
    --export="${TEST_EXPORT}" \
    "${SCRIPT_DIR}/_unit_test_runner.sh")"

log_info "Unit test job submitted: ${TEST_JOB_ID} (depends on build: ${BUILD_JOB_ID})"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log_info "=== Pipeline submitted ==="
log_info "  Build:      ${BUILD_JOB_ID}"
log_info "  Unit Tests: ${TEST_JOB_ID}"

echo "${TEST_JOB_ID}"
