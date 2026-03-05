#!/usr/bin/env bash
#SBATCH --job-name=models-post-commit
#SBATCH --partition=build
#SBATCH --time=00:30:00
#
# GHA source: .github/workflows/models-post-commit.yaml (orchestrator portion)
# Orchestrator: submits all model test suites (unit, e2e, sweep) across tiers.
# Does not run tests itself — fans out to tier-specific orchestrators which in
# turn submit array jobs against the worker scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/matrix.sh"
source "${SCRIPT_DIR}/workflows/_helpers/submit_dependent.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

parse_common_args "$@"

WORKFLOW_DIR="${SCRIPT_DIR}/workflows"

log_info "Submitting models post-commit pipeline"

# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------
T1_UNIT=$(submit_after "" "${WORKFLOW_DIR}/models-t1-unit-tests.sh" \
    --export=ALL,PIPELINE_ID="${PIPELINE_ID}")
T1_E2E=$(submit_after "" "${WORKFLOW_DIR}/models-t1-e2e-tests.sh" \
    --export=ALL,PIPELINE_ID="${PIPELINE_ID}")
T1_SWEEP=$(submit_after "" "${WORKFLOW_DIR}/models-t1-sweep-tests.sh" \
    --export=ALL,PIPELINE_ID="${PIPELINE_ID}")

# ---------------------------------------------------------------------------
# Tier 2
# ---------------------------------------------------------------------------
T2_UNIT=$(submit_after "" "${WORKFLOW_DIR}/models-t2-unit-tests.sh" \
    --export=ALL,PIPELINE_ID="${PIPELINE_ID}")
T2_E2E=$(submit_after "" "${WORKFLOW_DIR}/models-t2-e2e-tests.sh" \
    --export=ALL,PIPELINE_ID="${PIPELINE_ID}")
T2_SWEEP=$(submit_after "" "${WORKFLOW_DIR}/models-t2-sweep-tests.sh" \
    --export=ALL,PIPELINE_ID="${PIPELINE_ID}")

# ---------------------------------------------------------------------------
# Tier 3
# ---------------------------------------------------------------------------
T3_UNIT=$(submit_after "" "${WORKFLOW_DIR}/models-t3-unit-tests.sh" \
    --export=ALL,PIPELINE_ID="${PIPELINE_ID}")
T3_E2E=$(submit_after "" "${WORKFLOW_DIR}/models-t3-e2e-tests.sh" \
    --export=ALL,PIPELINE_ID="${PIPELINE_ID}")

log_info "Submitted models post-commit suites:"
log_info "  t1-unit:  ${T1_UNIT}"
log_info "  t1-e2e:   ${T1_E2E}"
log_info "  t1-sweep: ${T1_SWEEP}"
log_info "  t2-unit:  ${T2_UNIT}"
log_info "  t2-e2e:   ${T2_E2E}"
log_info "  t2-sweep: ${T2_SWEEP}"
log_info "  t3-unit:  ${T3_UNIT}"
log_info "  t3-e2e:   ${T3_E2E}"
log_info "All models post-commit jobs submitted"
