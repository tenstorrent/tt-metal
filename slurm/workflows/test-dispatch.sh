#!/usr/bin/env bash
#SBATCH --job-name=test-dispatch
#SBATCH --partition=build
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#
# Test job dispatch validation: verify the Slurm CI infrastructure
# can correctly submit, track, and clean up jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_config env

require_env PIPELINE_ID

log_info "=== Test dispatch starting ==="
log_info "  Pipeline: ${PIPELINE_ID}"

log_info "Validating artifact directory"
ARTIFACT_DIR="${SLURM_CI_ARTIFACT_BASE}/${PIPELINE_ID}"
mkdir -p "${ARTIFACT_DIR}/test-dispatch"
echo "dispatch-test-$(date -u '+%s')" > "${ARTIFACT_DIR}/test-dispatch/marker"

log_info "Validating Slurm environment"
for var in SLURM_JOB_ID SLURM_JOB_NAME; do
    if [[ -n "${!var:-}" ]]; then
        log_info "  ${var}=${!var}"
    else
        log_warn "  ${var} not set (expected in Slurm context)"
    fi
done

log_info "Validating shared filesystem access"
if [[ -d "${SLURM_CI_ARTIFACT_BASE}" ]]; then
    log_info "  Artifact base accessible: ${SLURM_CI_ARTIFACT_BASE}"
else
    log_error "  Artifact base NOT accessible: ${SLURM_CI_ARTIFACT_BASE}"
    exit 1
fi

log_info "Validating log directory"
LOG_DIR="${LOG_BASE}/${SLURM_JOB_NAME:-test-dispatch}"
mkdir -p "${LOG_DIR}" 2>/dev/null || log_warn "Could not create log dir: ${LOG_DIR}"

rm -f "${ARTIFACT_DIR}/test-dispatch/marker"
rmdir "${ARTIFACT_DIR}/test-dispatch" 2>/dev/null || true

log_info "=== Test dispatch complete ==="
