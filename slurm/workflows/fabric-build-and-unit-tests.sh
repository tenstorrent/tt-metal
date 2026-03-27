#!/usr/bin/env bash
#SBATCH --job-name=fabric-build-and-unit-tests
#SBATCH --partition=build
#SBATCH --time=00:30:00

# Two-stage workflow: build on the build partition, then test on wh-n150.
# This script acts as an orchestrator that submits both stages.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/artifacts.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/lib/matrix.sh"
source "${SCRIPT_DIR}/workflows/_helpers/submit_dependent.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

parse_common_args "$@"
resolve_docker_image dev

WORKFLOW_DIR="${SCRIPT_DIR}/workflows"

log_info "Submitting fabric build-and-unit-tests pipeline"

# Stage 1: Build on the build partition
BUILD_JOB=$(sbatch --parsable \
    --job-name=fabric-build \
    --partition=build \
    --time=01:00:00 \
    --output="${LOG_DIR}/%x-%j.out" \
    --export=ALL,PIPELINE_ID="${PIPELINE_ID}" \
    --wrap="
        set -euo pipefail
        source ${SCRIPT_DIR}/lib/common.sh
        source ${SCRIPT_DIR}/lib/docker.sh
        source ${SCRIPT_DIR}/lib/setup_job.sh
        source ${SCRIPT_DIR}/lib/cleanup.sh
        source ${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh
        resolve_docker_image ci-build
        setup_job
        trap 'cleanup_job --exit-code \$?' EXIT
        docker_run \"\$DOCKER_IMAGE\" 'cmake --preset fabric && cmake --build --preset fabric'
    ")

log_info "Build job submitted: ${BUILD_JOB}"

# Stage 2: Unit tests on wh-n150 (depend on build)
TEST_JOB=$(sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --job-name=fabric-unit-tests \
    --partition=wh-n150 \
    --time=02:00:00 \
    --output="${LOG_DIR}/%x-%j.out" \
    --export=ALL,PIPELINE_ID="${PIPELINE_ID}",BUILD_ARTIFACT=1 \
    --wrap="
        set -euo pipefail
        source ${SCRIPT_DIR}/lib/common.sh
        source ${SCRIPT_DIR}/lib/docker.sh
        source ${SCRIPT_DIR}/lib/setup_job.sh
        source ${SCRIPT_DIR}/lib/cleanup.sh
        source ${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh
        resolve_docker_image dev
        setup_job
        trap 'cleanup_job --exit-code \$?' EXIT
        docker_run \"\$DOCKER_IMAGE\" 'pytest tests/tt_fabric/unit -x --timeout=600'
    ")

log_info "Test job submitted: ${TEST_JOB} (depends on build ${BUILD_JOB})"
log_info "Fabric build-and-unit-tests pipeline submitted"
