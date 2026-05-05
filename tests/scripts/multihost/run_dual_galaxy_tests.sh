#!/bin/bash
set -eo pipefail

# Exit immediately if ARCH_NAME is not set or empty
if [ -z "${ARCH_NAME}" ]; then
  echo "Error: ARCH_NAME is not set. Exiting." >&2
  exit 1
fi

validate_dual_hosts_csv() {
  local hosts_csv="$1"
  local parsed_hosts
  IFS=',' read -r -a parsed_hosts <<< "$hosts_csv"
  if [[ ${#parsed_hosts[@]} -ne 2 ]]; then
    echo "Error: dual hosts must contain exactly 2 comma-separated hosts (got '${hosts_csv}')." >&2
    return 1
  fi

  local host0="${parsed_hosts[0]//[[:space:]]/}"
  local host1="${parsed_hosts[1]//[[:space:]]/}"
  if [[ -z "${host0}" || -z "${host1}" ]]; then
    echo "Error: dual hosts must not contain empty hostnames (got '${hosts_csv}')." >&2
    return 1
  fi

  echo "${host0},${host1}"
}

resolve_dual_hosts() {
  local hostfile="$1"
  if [[ -n "${DUAL_GALAXY_HOSTS:-}" ]]; then
    validate_dual_hosts_csv "${DUAL_GALAXY_HOSTS}"
    return $?
  fi

  if [[ ! -f "${hostfile}" ]]; then
    echo "Error: hostfile '${hostfile}' does not exist. Set DUAL_GALAXY_HOSTS=host0,host1 to choose dual hosts explicitly." >&2
    return 1
  fi

  local inferred_hosts
  inferred_hosts="$(
    awk '!/^#/ && NF {print $1}' "${hostfile}" | awk '
      NR == 1 {h1 = $1}
      NR == 2 {h2 = $1}
      END {
        if (NR < 2) {
          exit 1
        }
        printf "%s,%s", h1, h2
      }
    '
  )" || {
    echo "Error: unable to infer two hosts from '${hostfile}'. Set DUAL_GALAXY_HOSTS=host0,host1 to choose dual hosts explicitly." >&2
    return 1
  }

  validate_dual_hosts_csv "${inferred_hosts}"
}

run_dual_galaxy_unit_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_dual_galaxy_unit_tests"

  local rankfile="${DUAL_GALAXY_RANKFILE:-/etc/mpirun/rankfile}"
  local hostfile="${DUAL_GALAXY_HOSTFILE:-/etc/mpirun/hostfile}"
  if [[ ! -f "${rankfile}" ]]; then
    echo "Error: rankfile '${rankfile}' does not exist. Set DUAL_GALAXY_RANKFILE to the rankfile for your selected dual hosts." >&2
    exit 1
  fi
  local hosts
  hosts="$(resolve_dual_hosts "${hostfile}")" || exit 1

  local mpi_args_base="--map-by rankfile:file=${rankfile}"
  local tcp_interface="cnx1"

  local mpi_args="--host $hosts $mpi_args_base"

  local mpirun_args_base="$mpi_args_base --mca btl self,tcp --mca btl_tcp_if_include ${tcp_interface} --tag-output"
  local mpirun_args="--host $hosts $mpirun_args_base"
  local mesh_graph="tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto"

  echo "LOG_METAL: Using dual hosts '${hosts}'"
  echo "LOG_METAL: Using rankfile '${rankfile}'"

  mpirun-ulfm $mpirun_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/test/tt_metal/tt_fabric/test_physical_discovery ; fail+=$?
  mpirun-ulfm $mpirun_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/tools/scaleout/run_cluster_validation --print-connectivity --send-traffic --hard-fail ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts" ./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.ReportIntermeshLinks" ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDualGalaxyControlPlaneInit" ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts" ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="MultiHost.TestDualGalaxyFabric2DSanity" ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts" ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_dual_galaxy_fabric_2d_sanity.yaml ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts" pytest -svv "tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x8_dual_galaxy[wormhole_b0-dram-dram-DataType.BFLOAT16-None-1-s2-7168-8-256-32-1-8x8_grid-False-fabric_2d]" ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts" pytest -svv "tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x8_dual_galaxy[wormhole_b0-dram-dram-DataType.BFLOAT16-None-1-s2-7168-8-256-32-1-8x8_grid-False-fabric_1d_line]" ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts" pytest -svv "tests/nightly/tg/ccl/test_all_to_all_combine_6U.py::test_all_to_all_combine_8x8_dual_galaxy[wormhole_b0-dram-dram-DataType.BFLOAT16-None-num_links_1-2-sparse-s2-7168-8-256-32-axis_1-8x8_grid-False-fabric_1d_line]" ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts" pytest -svv "tests/nightly/tg/ccl/test_minimal_reduce_scatter_async.py::test_reduce_scatter_async_big_mesh" ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts" pytest -svv "tests/nightly/tg/ccl/test_minimal_all_gather_async.py::test_all_gather_async_big_mesh" ; fail+=$?
  tt-run --tcp-interface ${tcp_interface} --mesh-graph-descriptor "$mesh_graph" --hosts "$hosts" pytest -svv "tests/ttnn/unit_tests/base_functionality/test_multi_host_clusters.py::test_dual_galaxy_mesh_device_trace" ; fail+=$?

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
