#!/usr/bin/env bash
#SBATCH --job-name=models-t3-unit-tests
#SBATCH --partition=build
#SBATCH --time=00:30:00
#
# GHA source: .github/workflows/models-unit-tests-impl.yaml (tier=3 caller)
# Orchestrator: loads the tier-3 unit test matrix from config and submits
# models-unit-tests.sh as a Slurm job array.
#
# The matrix config is generated from tests/pipeline_reorg/models_unit_tests.yaml
# filtered to tier=3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/matrix.sh"
source "${SCRIPT_DIR}/workflows/_helpers/submit_dependent.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

parse_common_args "$@"

WORKFLOW_DIR="${SCRIPT_DIR}/workflows"
MATRIX_CONFIG="${SCRIPT_DIR}/config/matrices/models-t3-unit.json"

log_info "Submitting models tier-3 unit tests"

if [[ -f "${MATRIX_CONFIG}" ]]; then
    MATRIX_FILE=$(create_matrix_file "$(< "${MATRIX_CONFIG}")")
else
    log_fatal "Matrix config not found: ${MATRIX_CONFIG}"
fi

JOB_ID=$(submit_job_array "${WORKFLOW_DIR}/models-unit-tests.sh" "${MATRIX_FILE}")
log_info "Submitted models-t3-unit-tests array: ${JOB_ID}"
