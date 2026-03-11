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

RANK_BINDING="tests/tt_metal/distributed/config/dual_t3k_rank_bindings.yaml"
STRICT_RANK_BINDING="tests/tt_metal/distributed/config/dual_t3k_strict_connection_rank_bindings.yaml"
MPI_ARGS="--hostfile ${_alloc}/hostfile.txt"
MPIRUN_ARGS="${MPI_ARGS} --mca btl_tcp_if_exclude docker0,lo"

run_test "mpirun ${MPIRUN_ARGS} -x TT_METAL_HOME -x LD_LIBRARY_PATH \
    ./build/test/tt_metal/tt_fabric/test_physical_discovery"

run_test "mpirun ${MPIRUN_ARGS} -x TT_METAL_HOME -x LD_LIBRARY_PATH \
    ./build/tools/scaleout/run_cluster_validation --print-connectivity --send-traffic --hard-fail"

run_test "tt-run --rank-binding ${RANK_BINDING} --mpi-args '${MPI_ARGS}' \
    ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric \
    --test_config tests/tt_metal/perf_microbenchmark/routing/test_dual_t3k.yaml"

run_test "tt-run --rank-binding ${RANK_BINDING} --mpi-args '${MPI_ARGS}' \
    ./build/test/tt_metal/multi_host_fabric_tests"

run_test "tt-run --rank-binding ${RANK_BINDING} --mpi-args '${MPI_ARGS}' \
    ./build/test/tt_metal/test_mesh_socket_main \
    --test_config tests/tt_metal/multihost/fabric_tests/mesh_socket_dual_t3k.yaml"

run_test "tt-run --rank-binding ${STRICT_RANK_BINDING} --mpi-args '${MPI_ARGS}' \
    ./build/test/tt_metal/multi_host_fabric_tests"
