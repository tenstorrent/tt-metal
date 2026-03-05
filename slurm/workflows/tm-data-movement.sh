#!/usr/bin/env bash
#SBATCH --job-name=tm-data-movement
#SBATCH --partition=build
#SBATCH --time=00:30:00
#SBATCH --output=/weka/ci/logs/%x/%j.log
#SBATCH --error=/weka/ci/logs/%x/%j.err

# Orchestrator: builds artifacts then fans out to data-movement unit and perf
# test suites. Equivalent to .github/workflows/tm-data-movement-wrapper.yaml.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/matrix.sh"
source "${SCRIPT_DIR}/workflows/_helpers/submit_dependent.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

parse_common_args "$@"

WORKFLOW_DIR="${SCRIPT_DIR}/workflows"

log_info "Submitting TM data-movement pipeline"

# ---------------------------------------------------------------------------
# Stage 1: Build artifact
# ---------------------------------------------------------------------------
BUILD_JOB=$(submit_after "" "${WORKFLOW_DIR}/build-artifact.sh" \
    --export=ALL,PIPELINE_ID="${PIPELINE_ID}")

log_info "Build job: ${BUILD_JOB}"

# ---------------------------------------------------------------------------
# Stage 2: Test suites (depend on successful build)
# ---------------------------------------------------------------------------
UNIT_JOB=$(submit_after "${BUILD_JOB}" "${WORKFLOW_DIR}/tm-data-movement-unit.sh")
PERF_JOB=$(submit_after "${BUILD_JOB}" "${WORKFLOW_DIR}/tm-data-movement-perf.sh")

log_info "Submitted TM data-movement test suites:"
log_info "  unit: ${UNIT_JOB}"
log_info "  perf: ${PERF_JOB}"
log_info "All TM data-movement jobs submitted"
