#!/bin/bash
set -eo pipefail

# Exit immediately if ARCH_NAME is not set or empty
if [ -z "${ARCH_NAME}" ]; then
  echo "Error: ARCH_NAME is not set. Exiting." >&2
  exit 1
fi

run_quad_galaxy_fabric_sanity_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_quad_galaxy_fabric_sanity_tests"

  source python_env/bin/activate

  local mpi_args_base="--map-by rankfile:file=/etc/mpirun/rankfile --mca btl self,tcp --mca btl_tcp_if_include cnx1 --tag-output"
  #local mpi_args="--host g05glx01,g05glx02,g05glx03,g05glx04 $mpi_args_base"
  local mpi_args="--host g05glx04,g05glx03,g05glx02,g05glx01 $mpi_args_base"
  local rank_binding="tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml"

  # TODO: Currently failing
  #mpirun-ulfm $mpi_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/test/tt_metal/tt_fabric/test_physical_discovery ; fail+=$?

  # Cluster validation and connectivity tests
  echo "LOG_METAL: Running cluster validation tests"
  mpirun-ulfm $mpi_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/tools/scaleout/run_cluster_validation --print-connectivity --send-traffic --hard-fail ; fail+=$?

  # System health tests
  echo "LOG_METAL: Running system health tests"
  tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args" ./build/test/tt_metal/tt_fabric/test_system_health --gtest_filter="Cluster.ReportIntermeshLinks" ; fail+=$?

  # Multi-host cluster mesh device tests
  echo "LOG_METAL: Running multi-host cluster mesh device tests"
  tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args" bash -c "source ./python_env/bin/activate && pytest -svv \"tests/ttnn/unit_tests/base_functionality/test_multi_host_clusters.py::test_quad_galaxy_mesh_device_trace\"" ; fail+=$?

  # TODO: Currently failing on 1D/2D tests
  #tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args" bash -c "./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=\"MultiHost.TestQuadGalaxy*\"" ; fail+=$?

  # CCL all-to-all dispatch tests
  echo "LOG_METAL: Running CCL all-to-all dispatch tests"
  tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args" bash -c "source python_env/bin/activate && pytest -svv \"tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x16_quad_galaxy\"" ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_quad_galaxy_fabric_sanity_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_dual_galaxy_deepseek_tests() {
  echo "LOG_METAL: Running run_dual_galaxy_deepseek_tests"

  export MESH_DEVICE=DUAL
  source python_env/bin/activate

  local mpi_args_base="--map-by rankfile:file=/etc/mpirun/rankfile_01_02 --mca btl self,tcp --mca btl_tcp_if_include cnx1 --tag-output"
  local mpi_args="--host g05glx02,g05glx01 $mpi_args_base"
  local rank_binding="tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml"

  # Install DeepSeek requirements
  echo "LOG_METAL: Installing DeepSeek requirements"
  pip install -r models/demos/deepseek_v3/reference/deepseek/requirements.txt ; fail+=$?

  # Run DeepSeek module tests with tt-run wrapper (excluding unit tests as per existing workflow pattern)
  echo "LOG_METAL: Running DeepSeek module tests"
  tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args" bash -c "source python_env/bin/activate && pytest models/demos/deepseek_v3/tests"
}

run_quad_galaxy_deepseek_tests() {
  echo "LOG_METAL: Running run_quad_galaxy_deepseek_tests"

  export MESH_DEVICE=QUAD
  source python_env/bin/activate

  local mpi_args_base="--map-by rankfile:file=/etc/mpirun/rankfile --mca btl self,tcp --mca btl_tcp_if_include cnx1 --tag-output"
  local mpi_args="--host g05glx04,g05glx03,g05glx02,g05glx01 $mpi_args_base"
  local rank_binding="tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml"

  echo "LOG_METAL: Installing DeepSeek requirements"
  pip install -r models/demos/deepseek_v3/reference/deepseek/requirements.txt ; fail+=$?

  echo "LOG_METAL: Running DeepSeek module tests"
  tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args" bash -c "source python_env/bin/activate && pytest models/demos/deepseek_v3/tests"
}

run_quad_galaxy_tests() {
  run_quad_galaxy_fabric_sanity_tests
  run_dual_galaxy_deepseek_tests
  run_quad_galaxy_deepseek_tests
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

  run_quad_galaxy_tests

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
