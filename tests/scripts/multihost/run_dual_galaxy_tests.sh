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

  local mpi_args_base="--map-by rankfile:file=/etc/mpirun/rankfile"
  local tcp_interface="cnx1"
  local hosts="g10glx03,g10glx04"
  local hosts_reversed="g10glx04,g10glx03"
  local mpirun_args_base="$mpi_args_base --mca btl self,tcp --mca btl_tcp_if_include cnx1 --tag-output"
  local mpirun_args="--host $hosts $mpirun_args_base"
  local mesh_graph="tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto"

  mpirun-ulfm $mpirun_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/test/tt_metal/tt_fabric/test_physical_discovery ; fail+=$?
  # Physical discovery with launcher NOT in hosts (OpenMPI #11830 - would hang without P2P workaround)
  # Use hostfile from /etc/mpirun if available; galaxy dual has hardcoded hosts so create temp hostfile
  if [[ -f /etc/mpirun/hostfile ]]; then
    HOSTFILE=/etc/mpirun/hostfile ./tests/scripts/multihost/test_physical_discovery_launcher_not_in_hosts.sh ; fail+=$?
  else
    echo "g10glx03" > /tmp/hostfile_galaxy ; echo "g10glx04" >> /tmp/hostfile_galaxy
    HOSTFILE=/tmp/hostfile_galaxy ./tests/scripts/multihost/test_physical_discovery_launcher_not_in_hosts.sh ; fail+=$?
  fi
  mpirun-ulfm $mpirun_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/tools/scaleout/run_cluster_validation --print-connectivity --send-traffic --hard-fail ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts" ./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.ReportIntermeshLinks" ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDualGalaxyControlPlaneInit" ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDualGalaxyFabric2DSanity" ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts" ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_dual_galaxy_fabric_2d_sanity.yaml ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts_reversed" pytest -svv "tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x8_dual_galaxy[wormhole_b0-dram-dram-DataType.BFLOAT16-None-1-s2-7168-8-256-32-1-8x8_grid-False-fabric_2d]" ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts_reversed" pytest -svv "tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x8_dual_galaxy[wormhole_b0-dram-dram-DataType.BFLOAT16-None-1-s2-7168-8-256-32-1-8x8_grid-False-fabric_1d_line]" ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts_reversed" pytest -svv "tests/nightly/tg/ccl/test_all_to_all_combine_6U.py::test_all_to_all_combine_8x8_dual_galaxy[wormhole_b0-dram-dram-DataType.BFLOAT16-None-num_links_1-2-sparse-s2-7168-8-256-32-axis_1-8x8_grid-False-fabric_1d_line]" ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts_reversed" pytest -svv "tests/nightly/tg/ccl/test_minimal_reduce_scatter_async.py::test_reduce_scatter_async_big_mesh" ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts_reversed" pytest -svv "tests/nightly/tg/ccl/test_minimal_all_gather_async.py::test_all_gather_async_big_mesh" ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts_reversed" pytest -svv "tests/ttnn/unit_tests/base_functionality/test_multi_host_clusters.py::test_dual_galaxy_mesh_device_trace" ; fail+=$?

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

  pytest models/demos/vision/classification/resnet50/ttnn_resnet/tests/test_perf_e2e_resnet50.py ; fail+=$?

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
