#!/bin/bash
set -eo pipefail

# Exit immediately if ARCH_NAME is not set or empty
if [ -z "${ARCH_NAME}" ]; then
  echo "Error: ARCH_NAME is not set. Exiting." >&2
  exit 1
fi

run_dual_galaxy_unit_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_dual_galaxy_unit_tests"

  source python_env/bin/activate

  local mpi_args_base="--map-by rankfile:file=/etc/mpirun/rankfile --mca btl self,tcp --mca btl_tcp_if_include cnx1 --tag-output"
  local mpi_args="--host g10glx03,g10glx04 $mpi_args_base"
  local mpi_args_reversed="--host g10glx04,g10glx03 $mpi_args_base"
  local rank_binding="tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml"

  mpirun-ulfm $mpi_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/test/tt_metal/tt_fabric/test_physical_discovery ; fail+=$?
  mpirun-ulfm $mpi_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/tools/scaleout/run_cluster_validation --print-connectivity --send-traffic --hard-fail ; fail+=$?
  tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args" ./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.ReportIntermeshLinks" ; fail+=$?
  tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDualGalaxyControlPlaneInit" ; fail+=$?
  tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDualGalaxyFabric2DSanity" ; fail+=$?
  tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args" ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_dual_galaxy_fabric_2d_sanity.yaml ; fail+=$?
  tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args_reversed" bash -c "source ./python_env/bin/activate && pytest -svv \"tests/ttnn/unit_tests/operations/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_no_trace[silicon_arch_name=wormhole_b0-dram-l1-dtype=DataType.BFLOAT16-topology=None-num_links=1-s2-hidden_size=7168-select_experts_k=8-experts=256-batches_per_device=32-cluster_axis=1-8x8_grid-trace_mode=False-fabric_2d]\"" ; fail+=$?
  tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args_reversed" bash -c "source ./python_env/bin/activate && pytest -svv \"tests/ttnn/unit_tests/operations/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_no_trace[silicon_arch_name=wormhole_b0-dram-l1-dtype=DataType.BFLOAT16-topology=None-num_links=1-s2-hidden_size=7168-select_experts_k=8-experts=256-batches_per_device=32-cluster_axis=1-8x8_grid-trace_mode=False-fabric_1d_line]\"" ; fail+=$?
  tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args_reversed" bash -c "source ./python_env/bin/activate && pytest -svv \"tests/nightly/tg/ccl/test_minimal_reduce_scatter_async.py::test_reduce_scatter_async_big_mesh\"" ; fail+=$?
  tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args_reversed" bash -c "source ./python_env/bin/activate && pytest -svv \"tests/nightly/tg/ccl/test_minimal_all_gather_async.py::test_all_gather_async_big_mesh\"" ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_dual_galaxy_unit_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_dual_galaxy_resnet50_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_dual_galaxy_resnet50_tests"

  pytest models/demos/ttnn_resnet/tests/test_perf_e2e_resnet50.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_dual_galaxy_resnet50_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_dual_galaxy_tests() {
  run_dual_galaxy_unit_tests
  # TODO: #30155 - Enable the test when hardware hang is addressed.
  # run_dual_galaxy_resnet50_tests
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

  run_dual_galaxy_tests

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
