#!/usr/bin/env bash
#SBATCH --job-name=fabric-multihost-exabox
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
TCP_INTERFACE="${TCP_INTERFACE:-cnx1}"
MPIRUN_ARGS="${MPI_ARGS} --tag-output --mca btl self,tcp --mca btl_tcp_if_include ${TCP_INTERFACE}"

run_test "mpirun ${MPIRUN_ARGS} -x TT_METAL_HOME -x LD_LIBRARY_PATH -x ARCH_NAME \
    ./build/test/tt_metal/tt_fabric/test_physical_discovery"

run_test "mpirun ${MPIRUN_ARGS} -x TT_METAL_HOME -x LD_LIBRARY_PATH -x ARCH_NAME \
    ./build/tools/scaleout/run_cluster_validation --print-connectivity --send-traffic --hard-fail"

run_test "tt-run --tcp-interface ${TCP_INTERFACE} --rank-binding ${RANK_BINDING} --mpi-args '${MPI_ARGS}' \
    ./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter='Cluster.ReportIntermeshLinks'"

run_test "tt-run --tcp-interface ${TCP_INTERFACE} --rank-binding ${RANK_BINDING} --mpi-args '${MPI_ARGS}' \
    ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter='MultiHost.TestDualGalaxyControlPlaneInit'"

run_test "tt-run --tcp-interface ${TCP_INTERFACE} --rank-binding ${RANK_BINDING} --mpi-args '${MPI_ARGS}' \
    ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter='MultiHost.TestDualGalaxyFabric2DSanity'"
