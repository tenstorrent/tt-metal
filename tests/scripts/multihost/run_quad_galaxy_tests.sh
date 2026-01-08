#!/bin/bash
set -eo pipefail

# Exit immediately if ARCH_NAME is not set or empty
if [ -z "${ARCH_NAME}" ]; then
  echo "Error: ARCH_NAME is not set. Exiting." >&2
  exit 1
fi

run_quad_galaxy_unit_tests() {
  fail=0

  source python_env/bin/activate

  local mpi_args_base="--map-by rankfile:file=/etc/mpirun/rankfile --mca btl self,tcp --mca btl_tcp_if_include cnx1 --tag-output"
  local mpi_args="--host g05glx04,g05glx03,g05glx02,g05glx01 $mpi_args_base"
  local rank_binding="tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml"
  local descriptor_path="${DESCRIPTOR_PATH:-/etc/mpirun}"

  # TODO: Currently failing
  #mpirun-ulfm $mpi_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/test/tt_metal/tt_fabric/test_physical_discovery ; fail+=$?

  mpirun-ulfm $mpi_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/tools/scaleout/run_cluster_validation --send-traffic --cabling-descriptor-path ${descriptor_path}/cabling_descriptor.textproto --deployment-descriptor-path ${descriptor_path}/deployment_descriptor.textproto ; fail+=$?

  tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args" bash -c "source ./python_env/bin/activate && pytest -svv \"tests/ttnn/unit_tests/base_functionality/test_multi_host_clusters.py::test_quad_galaxy_mesh_device_trace\"" ; fail+=$?

  # TODO: Currently failing on 1D/2D tests
  #tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args" bash -c "./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=\"MultiHost.TestQuadGalaxy*\"" ; fail+=$?

  tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args" bash -c "source python_env/bin/activate && pytest -svv \"tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x16_quad_galaxy\"" ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_dual_galaxy_deepseekv3_tests_on_quad_galaxy() {
    fail=0

    # Run dual galaxy tests on quad galaxy since this is the only available machine
    local RANK_BINDING_YAML="tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml"
    local HOSTS="g05glx01,g05glx02"
    local RANKFILE=/etc/mpirun/rankfile_g05glx01_g05glx02

    if ! test -f "$RANKFILE"; then
        echo "File '$RANKFILE' does not exist."
        exit 1
    fi
    if ! test -f "$RANK_BINDING_YAML"; then
        echo "File '$RANK_BINDING_YAML' does not exist."
        exit 1
    fi

    local DEEPSEEK_V3_HF_MODEL="/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528"
    local DEEPSEEK_V3_CACHE="/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI"
    local MESH_DEVICE="DUAL"

    local TEST_CASE="pytest -svvv models/demos/deepseek_v3/tests"

    tt-run --rank-binding "$RANK_BINDING_YAML" \
        --mpi-args "--host $HOSTS --map-by rankfile:file=$RANKFILE --mca btl self,tcp --mca btl_tcp_if_include cnx1 --bind-to none --output-filename logs/mpi_job --tag-output" \
        bash -c "source ./python_env/bin/activate && export DEEPSEEK_V3_HF_MODEL=$DEEPSEEK_V3_HF_MODEL && export DEEPSEEK_V3_CACHE=$DEEPSEEK_V3_CACHE && export MESH_DEVICE=$MESH_DEVICE && $TEST_CASE" ; fail+=$?

    # Run test_demo_dual test on DUAL galaxy setup
    local TEST_DEMO_DUAL="pytest -svvv models/demos/deepseek_v3/demo/test_demo_dual.py::test_demo_dual"

    tt-run --rank-binding "$RANK_BINDING_YAML" \
        --mpi-args "--host $HOSTS --map-by rankfile:file=$RANKFILE --mca btl self,tcp --mca btl_tcp_if_include cnx1 --bind-to none --output-filename logs/mpi_job --tag-output" \
        bash -c "source ./python_env/bin/activate && export DEEPSEEK_V3_HF_MODEL=$DEEPSEEK_V3_HF_MODEL && export DEEPSEEK_V3_CACHE=$DEEPSEEK_V3_CACHE && export MESH_DEVICE=$MESH_DEVICE && $TEST_DEMO_DUAL" ; fail+=$?

    if [[ $fail -ne 0 ]]; then
        exit 1
    fi
}


run_quad_galaxy_tests() {
  run_quad_galaxy_unit_tests
  run_dual_galaxy_deepseekv3_tests_on_quad_galaxy
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
