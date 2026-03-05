#!/usr/bin/env bash
#SBATCH --job-name=blackhole-post-commit
#SBATCH --partition=build
#SBATCH --time=00:30:00
#SBATCH --output=/weka/ci/logs/%x/%j.log
#SBATCH --error=/weka/ci/logs/%x/%j.err

# Orchestrator: builds artifacts then fans out to all BH test suites.
# Equivalent to .github/workflows/blackhole-post-commit.yaml.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/matrix.sh"
source "${SCRIPT_DIR}/workflows/_helpers/submit_dependent.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

parse_common_args "$@"

WORKFLOW_DIR="${SCRIPT_DIR}/workflows"

log_info "Submitting blackhole post-commit pipeline"

# ---------------------------------------------------------------------------
# Stage 1: Build artifact
# ---------------------------------------------------------------------------
BUILD_JOB=$(submit_after "" "${WORKFLOW_DIR}/build-artifact.sh" \
    --export=ALL,PIPELINE_ID="${PIPELINE_ID}",ARCH_NAME=blackhole)

log_info "Build job: ${BUILD_JOB}"

# ---------------------------------------------------------------------------
# Stage 2: Test suites (all depend on successful build)
# ---------------------------------------------------------------------------
DEMO_JOB=$(submit_after "${BUILD_JOB}" "${WORKFLOW_DIR}/blackhole-demo-tests.sh")
E2E_JOB=$(submit_after "${BUILD_JOB}" "${WORKFLOW_DIR}/blackhole-e2e-tests.sh")
GRID_JOB=$(submit_after "${BUILD_JOB}" "${WORKFLOW_DIR}/blackhole-grid-override-tests.sh")
MULTI_CARD_JOB=$(submit_after "${BUILD_JOB}" "${WORKFLOW_DIR}/blackhole-multi-card-unit-tests.sh")

TTNN_JOB=$(submit_after "${BUILD_JOB}" "${WORKFLOW_DIR}/ttnn-post-commit.sh" \
    --export=ALL,PIPELINE_ID="${PIPELINE_ID}",ARCH_NAME=blackhole)
OPS_JOB=$(submit_after "${BUILD_JOB}" "${WORKFLOW_DIR}/ops-post-commit.sh" \
    --export=ALL,PIPELINE_ID="${PIPELINE_ID}",ARCH_NAME=blackhole)
MODELS_JOB=$(submit_after "${BUILD_JOB}" "${WORKFLOW_DIR}/models-post-commit.sh" \
    --export=ALL,PIPELINE_ID="${PIPELINE_ID}",ARCH_NAME=blackhole)
UMD_JOB=$(submit_after "${BUILD_JOB}" "${WORKFLOW_DIR}/umd-unit-tests.sh" \
    --export=ALL,PIPELINE_ID="${PIPELINE_ID}",ARCH_NAME=blackhole)
PROFILER_JOB=$(submit_after "${BUILD_JOB}" "${WORKFLOW_DIR}/run-profiler-regression.sh" \
    --export=ALL,PIPELINE_ID="${PIPELINE_ID}",ARCH_NAME=blackhole)
TT_CNN_JOB=$(submit_after "${BUILD_JOB}" "${WORKFLOW_DIR}/tt-cnn-post-commit.sh" \
    --export=ALL,PIPELINE_ID="${PIPELINE_ID}",ARCH_NAME=blackhole)

log_info "Submitted blackhole test suites:"
log_info "  demo:       ${DEMO_JOB}"
log_info "  e2e:        ${E2E_JOB}"
log_info "  grid:       ${GRID_JOB}"
log_info "  multi-card: ${MULTI_CARD_JOB}"
log_info "  ttnn:       ${TTNN_JOB}"
log_info "  ops:        ${OPS_JOB}"
log_info "  models:     ${MODELS_JOB}"
log_info "  umd:        ${UMD_JOB}"
log_info "  profiler:   ${PROFILER_JOB}"
log_info "  tt-cnn:     ${TT_CNN_JOB}"
log_info "All blackhole post-commit jobs submitted"
