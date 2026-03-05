#!/usr/bin/env bash
# build-all-docker-images.sh - Orchestrator that submits build-docker-artifact
# jobs for all platforms (Ubuntu 22.04, Ubuntu 24.04, ManyLinux) in parallel.
#
# Not itself an sbatch script — run directly from submit.sh or CI.
#
# Mirrors: .github/workflows/build-all-docker-images.yaml
#
# Submits up to 3 Slurm job arrays:
#   1. Ubuntu 22.04 images (ci-build, ci-test, dev, basic-dev, basic-ttnn)
#   2. Ubuntu 24.04 images (ci-build, ci-test, dev, basic-dev, basic-ttnn)
#   3. ManyLinux image
#
# Each array uses build-docker-artifact.sh with platform-specific env vars.
# Image-exists checks happen inside each array task (idempotent builds).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source "${SCRIPT_DIR}/_helpers/submit_dependent.sh"
source_config env

require_env PIPELINE_ID

BUILD_SCRIPT="${SCRIPT_DIR}/build-docker-artifact.sh"

log_info "=== Submitting Docker image builds for all platforms ==="
log_info "  Pipeline: ${PIPELINE_ID}"
log_info "  Script:   ${BUILD_SCRIPT}"

declare -a ALL_JOB_IDS=()

# ---------------------------------------------------------------------------
# Ubuntu 22.04 images (array tasks 0-4: ci-build, ci-test, dev, basic-dev, basic-ttnn)
# ---------------------------------------------------------------------------
log_info "Submitting Ubuntu 22.04 image builds"
UBUNTU_2204_JOB_ID="$(sbatch \
    --parsable \
    --job-name="docker-ubuntu2204-${PIPELINE_ID}" \
    --partition=build \
    --time=02:00:00 \
    --cpus-per-task=8 \
    --mem=32G \
    --array=0-4 \
    --output="logs/docker-ubuntu2204-${PIPELINE_ID}-%A_%a.out" \
    --export="ALL,PIPELINE_ID=${PIPELINE_ID},PLATFORM=Ubuntu 22.04" \
    "${BUILD_SCRIPT}")"

log_info "Ubuntu 22.04 array submitted: JOBID=${UBUNTU_2204_JOB_ID}"
ALL_JOB_IDS+=("${UBUNTU_2204_JOB_ID}")

# ---------------------------------------------------------------------------
# Ubuntu 24.04 images (array tasks 0-4)
# ---------------------------------------------------------------------------
log_info "Submitting Ubuntu 24.04 image builds"
UBUNTU_2404_JOB_ID="$(sbatch \
    --parsable \
    --job-name="docker-ubuntu2404-${PIPELINE_ID}" \
    --partition=build \
    --time=02:00:00 \
    --cpus-per-task=8 \
    --mem=32G \
    --array=0-4 \
    --output="logs/docker-ubuntu2404-${PIPELINE_ID}-%A_%a.out" \
    --export="ALL,PIPELINE_ID=${PIPELINE_ID},PLATFORM=Ubuntu 24.04" \
    "${BUILD_SCRIPT}")"

log_info "Ubuntu 24.04 array submitted: JOBID=${UBUNTU_2404_JOB_ID}"
ALL_JOB_IDS+=("${UBUNTU_2404_JOB_ID}")

# ---------------------------------------------------------------------------
# ManyLinux image (single task: array index 5)
# ---------------------------------------------------------------------------
log_info "Submitting ManyLinux image build"
MANYLINUX_JOB_ID="$(sbatch \
    --parsable \
    --job-name="docker-manylinux-${PIPELINE_ID}" \
    --partition=build \
    --time=02:00:00 \
    --cpus-per-task=8 \
    --mem=32G \
    --array=5 \
    --output="logs/docker-manylinux-${PIPELINE_ID}-%A_%a.out" \
    --export="ALL,PIPELINE_ID=${PIPELINE_ID},PLATFORM=Ubuntu 22.04" \
    "${BUILD_SCRIPT}")"

log_info "ManyLinux build submitted: JOBID=${MANYLINUX_JOB_ID}"
ALL_JOB_IDS+=("${MANYLINUX_JOB_ID}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log_info "=== All Docker image builds submitted ==="
log_info "  Ubuntu 22.04: ${UBUNTU_2204_JOB_ID}"
log_info "  Ubuntu 24.04: ${UBUNTU_2404_JOB_ID}"
log_info "  ManyLinux:    ${MANYLINUX_JOB_ID}"

# Print colon-separated job IDs for use with collect()
IFS=':' ; echo "${ALL_JOB_IDS[*]}"
