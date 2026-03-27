#!/usr/bin/env bash
# release-build-test-publish.sh - Orchestrator: full release pipeline
#
# Submits: build -> package -> demo tests -> publish image -> cleanup
# Intended to be launched via: ./slurm/submit.sh release-build-test-publish

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source "${SCRIPT_DIR}/_helpers/submit_dependent.sh"

require_env PIPELINE_ID

log_info "=== Release pipeline orchestrator ==="
log_info "  Pipeline: ${PIPELINE_ID}"

# Step 1: Build artifact
BUILD_JOB=$(submit_after "" "${SCRIPT_DIR}/build-artifact.sh" \
    --partition=build)

# Step 2: Package release (after build)
PACKAGE_JOB=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/package-and-release.sh" \
    --partition=build)

# Step 3: Release demo tests (after package)
DEMO_JOB=$(submit_after "${PACKAGE_JOB}" "${SCRIPT_DIR}/release-demo-tests.sh" \
    --partition=wh-n150)

# Step 4: Verify tag (after build, can run in parallel with tests)
TAG_JOB=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/release-verify-tag.sh" \
    --partition=build)

# Step 5: Publish release image (after tests and tag verification pass)
PUBLISH_JOB=$(submit_after_multiple "${DEMO_JOB}:${TAG_JOB}" \
    "${SCRIPT_DIR}/publish-release-image.sh" \
    --partition=build)

# Step 6: Cleanup (runs after everything, regardless of status)
CLEANUP_JOB=$(submit_after_any "${PUBLISH_JOB}" "${SCRIPT_DIR}/release-cleanup.sh" \
    --partition=build)

log_info "=== Release pipeline submitted ==="
log_info "  Build:    ${BUILD_JOB}"
log_info "  Package:  ${PACKAGE_JOB}"
log_info "  Demo:     ${DEMO_JOB}"
log_info "  Tag:      ${TAG_JOB}"
log_info "  Publish:  ${PUBLISH_JOB}"
log_info "  Cleanup:  ${CLEANUP_JOB}"
