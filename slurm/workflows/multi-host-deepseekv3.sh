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

RANK_BINDING="tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml"
MPI_ARGS="--hostfile ${_alloc}/hostfile.txt"
TCP_INTERFACE="cnx1"

export MESH_DEVICE=DUAL
export DEEPSEEK_V3_HF_MODEL="${DEEPSEEK_V3_HF_MODEL:-/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528}"
export DEEPSEEK_V3_CACHE="${DEEPSEEK_V3_CACHE:-/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI}"

# Forward DeepSeek-specific env vars into Docker containers
DOCKER_EXTRA_ENV="${DOCKER_EXTRA_ENV:+${DOCKER_EXTRA_ENV}
}MESH_DEVICE=${MESH_DEVICE}
DEEPSEEK_V3_HF_MODEL=${DEEPSEEK_V3_HF_MODEL}
DEEPSEEK_V3_CACHE=${DEEPSEEK_V3_CACHE}"
export DOCKER_EXTRA_ENV

run_test "tt-run --tcp-interface ${TCP_INTERFACE} --rank-binding ${RANK_BINDING} --mpi-args '${MPI_ARGS}' \
    pytest -svvv models/demos/deepseek_v3/tests/unit --timeout=1800"

run_test "tt-run --tcp-interface ${TCP_INTERFACE} --rank-binding ${RANK_BINDING} --mpi-args '${MPI_ARGS}' \
    pytest -svvv models/demos/deepseek_v3/tests \
    --ignore=models/demos/deepseek_v3/tests/unit \
    --ignore=models/demos/deepseek_v3/tests/fused_op_unit_tests --timeout=1800"
