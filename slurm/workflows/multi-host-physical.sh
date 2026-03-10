#!/usr/bin/env bash
#SBATCH --job-name=multi-host-physical
#SBATCH --partition=exabox
#SBATCH --nodes=2
#SBATCH --time=03:00:00

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/artifacts.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/lib/multihost.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

parse_common_args "$@"
resolve_docker_image dev
setup_job

ALLOC_DIR="/tmp/multihost-phys-${PIPELINE_ID}-${SLURM_JOB_ID:-0}"
mkdir -p "${ALLOC_DIR}"

cleanup_multihost() {
    local rc=$?
    rm -rf "${ALLOC_DIR}"
    cleanup_job --exit-code "${rc}"
}
trap 'cleanup_multihost' EXIT

multihost_setup "${ALLOC_DIR}"

TEST_CMD="pytest tests/tt_metal/multihost/physical -x --timeout=1200 \
    --hostfile=${ALLOC_DIR}/hostfile.txt \
    --rankfile=${ALLOC_DIR}/rankfile.txt"

docker_run "$DOCKER_IMAGE" "${TEST_CMD}"
