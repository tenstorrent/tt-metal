#!/usr/bin/env bash
#SBATCH --job-name=multi-host-deepseekv3
#SBATCH --partition=exabox
#SBATCH --time=04:00:00

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/artifacts.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/lib/ttop.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

export BUILD_ARTIFACT=1

parse_common_args "$@"
resolve_docker_image dev
setup_job

ALLOC_NAME="deepseekv3-${PIPELINE_ID}-${SLURM_JOB_ID:-0}"
ALLOC_DIR="/tmp/ttop-${ALLOC_NAME}"
mkdir -p "${ALLOC_DIR}"

cleanup_multihost() {
    local rc=$?
    ttop_delete_environment "${ALLOC_NAME}" || true
    ttop_delete_allocation "${ALLOC_NAME}" || true
    rm -rf "${ALLOC_DIR}"
    cleanup_job --exit-code "${rc}"
}
trap 'cleanup_multihost' EXIT

ttop_create_allocation "${ALLOC_NAME}" "${SCRIPT_DIR}/config/specs/exabox-deepseek.yaml" 600
ttop_create_environment "${ALLOC_NAME}" "${DOCKER_IMAGE}"
ttop_configure_tt_run "${ALLOC_NAME}" "${ALLOC_DIR}"

TEST_CMD="pytest models/demos/deepseek_v3/tests/test_multihost.py -x --timeout=1800 \
    --hostfile=${ALLOC_DIR}/hostfile.txt \
    --rankfile=${ALLOC_DIR}/rankfile.txt"

docker_run -- "${TEST_CMD}"
