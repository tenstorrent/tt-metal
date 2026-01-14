#!/bin/bash
set -eo pipefail

# Exit immediately if ARCH_NAME is not set or empty
if [ -z "${ARCH_NAME}" ]; then
  echo "Error: ARCH_NAME is not set. Exiting." >&2
  exit 1
fi

run_dual_t3k_unit_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_dual_t3k_unit_tests"

  source python_env/bin/activate

  local mpirun_args="--hostfile /etc/mpirun/hostfile --mca btl_tcp_if_exclude docker0,lo"
  local mpi_args="$mpirun_args --tag-output"
  local rank_binding="tests/tt_metal/distributed/config/dual_t3k_rank_bindings.yaml"
  local strict_rank_binding="tests/tt_metal/distributed/config/dual_t3k_strict_connection_rank_bindings.yaml"

  mpirun $mpi_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/test/tt_metal/tt_fabric/test_physical_discovery ; fail+=$?
  mpirun $mpi_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/tools/scaleout/run_cluster_validation  --print-connectivity --send-traffic --hard-fail ; fail+=$?
  tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args" ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/perf_microbenchmark/routing/test_dual_t3k.yaml ; fail+=$?
  tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args" ./build/test/tt_metal/multi_host_fabric_tests ; fail+=$?
  tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args" ./build/test/tt_metal/test_mesh_socket_main --test_config tests/tt_metal/multihost/fabric_tests/mesh_socket_dual_t3k.yaml ; fail+=$?
  tt-run --rank-binding "$strict_rank_binding" --mpi-args "$mpi_args" ./build/test/tt_metal/multi_host_fabric_tests ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_dual_t3k_unit_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_dual_t3k_tests() {
  run_dual_t3k_unit_tests
}

fail=0
main() {
  # For CI pipeline - source func commands but don't execute tests if not invoked directly
  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "Script is being sourced, not executing main function"
    return 0
  fi

  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  # Run all tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  run_dual_t3k_tests

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
