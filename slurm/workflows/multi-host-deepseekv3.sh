#!/usr/bin/env bash
#SBATCH --job-name=multi-host-deepseekv3
#SBATCH --partition=exabox
#SBATCH --nodes=2
#SBATCH --time=04:00:00

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/docker.sh"
source "${SCRIPT_DIR}/lib/artifacts.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/lib/multihost.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_docker_image.sh"

export BUILD_ARTIFACT=1

parse_common_args "$@"
resolve_docker_image dev
setup_job

ALLOC_DIR="${ARTIFACT_DIR}/multihost-$(hostname -s)"
mkdir -p "${ALLOC_DIR}"

cleanup_multihost() {
    local rc=$?
    rm -rf "${ALLOC_DIR}"
    cleanup_job --exit-code "${rc}"
}
trap 'cleanup_multihost' EXIT

multihost_setup "${ALLOC_DIR}"

if [[ "${NO_DOCKER:-0}" == "1" ]]; then
    _alloc="${ALLOC_DIR}"
else
    _alloc="/artifacts/multihost-$(hostname -s)"
fi

TEST_CMD="pytest models/demos/deepseek_v3/tests/test_multihost.py -x --timeout=1800 \
    --hostfile=${_alloc}/hostfile.txt \
    --rankfile=${_alloc}/rankfile.txt"

run_test "${TEST_CMD}"
