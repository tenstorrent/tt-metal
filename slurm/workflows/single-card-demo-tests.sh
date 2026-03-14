#!/usr/bin/env bash
#SBATCH --job-name=single-card-demo-tests
#SBATCH --partition=wh-n150
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#
# GHA source: single-card demo test workflows
# Runs demo tests across N150/N300 cards. Uses a matrix file to map
# TASK_ID to specific demo + card type combinations.
#
# Environment overrides:
#   MATRIX_FILE - JSON matrix mapping TASK_ID -> {demo, card}
#   CARD_TYPE   - Override card type (default: determined by partition)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/artifacts.sh"
source "${SCRIPT_DIR}/lib/matrix.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

parse_common_args "$@"
resolve_workflow_docker_image dev

export BUILD_ARTIFACT=1
export INSTALL_WHEEL=1
setup_job
trap 'cleanup_job $?' EXIT

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [[ -n "${MATRIX_FILE:-}" ]]; then
    DEMO_NAME="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "demo")"
    CARD_TYPE="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "card" 2>/dev/null || echo "n150")"
    TIMEOUT="$(get_task_field "$MATRIX_FILE" "$TASK_ID" "timeout" 2>/dev/null || echo "30")"
else
    DEMO_NAME="demo_${TASK_ID}"
    CARD_TYPE="${CARD_TYPE:-n150}"
    TIMEOUT=30
fi
log_info "Running single-card demo test: ${DEMO_NAME} on ${CARD_TYPE} (task ${TASK_ID})"

export DOCKER_EXTRA_ENV="GTEST_OUTPUT=xml:generated/test_reports/"
export DOCKER_EXTRA_VOLUMES="${MLPERF_BASE}:${MLPERF_BASE}:ro"

docker_run "$DOCKER_IMAGE" "\
    pytest demos/${DEMO_NAME}/ \
        -x --timeout=900 \
        --junit-xml=generated/test_reports/demo_${TASK_ID}.xml
"
