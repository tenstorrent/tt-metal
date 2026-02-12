#!/bin/bash
set -eo pipefail

# Exit immediately if ARCH_NAME is not set or empty
if [ -z "${ARCH_NAME}" ]; then
  echo "Error: ARCH_NAME is not set. Exiting." >&2
  exit 1
fi

run_quad_galaxy_unit_tests() {
  fail=0

  # tt-run --tcp-interface handles tcp and tag flags
  local mpi_args_base="--map-by rankfile:file=/etc/mpirun/rankfile"
  local tcp_interface="cnx1"
  local mpi_host="--host g05glx04,g05glx03,g05glx02,g05glx01"
  local mpi_args="$mpi_host $mpi_args_base"

  local mpirun_args_base="$mpi_args_base --mca btl self,tcp --mca btl_tcp_if_include cnx1 --tag-output"
  local mpirun_args="$mpi_host $mpirun_args_base"

  local rank_binding="tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml"
  local descriptor_path="${DESCRIPTOR_PATH:-/etc/mpirun}"

  # TODO: Currently failing
  #mpirun-ulfm $mpi_run_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/test/tt_metal/tt_fabric/test_physical_discovery ; fail+=$?

  mpirun-ulfm $mpirun_args -x TT_METAL_HOME=$(pwd) -x LD_LIBRARY_PATH=$(pwd)/build/lib ./build/tools/scaleout/run_cluster_validation --send-traffic --cabling-descriptor-path ${descriptor_path}/cabling_descriptor.textproto --deployment-descriptor-path ${descriptor_path}/deployment_descriptor.textproto ; fail+=$?

  tt-run --tcp-interface $tcp_interface --rank-binding "$rank_binding" --mpi-args "$mpi_args" pytest -svv "tests/ttnn/unit_tests/base_functionality/test_multi_host_clusters.py::test_quad_galaxy_mesh_device_trace" ; fail+=$?

  # TODO: Currently failing on 1D/2D tests
  #tt-run --tcp-interface $tcp_interface --rank-binding "$rank_binding" --mpi-args "$mpi_args" bash -c "./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=\"MultiHost.TestQuadGalaxy*\"" ; fail+=$?

  tt-run --tcp-interface $tcp_interface --rank-binding "$rank_binding" --mpi-args "$mpi_args" pytest -svv "tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x16_quad_galaxy" ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

# Common setup for dual galaxy tests on quad galaxy
setup_dual_galaxy_env() {
    export RANK_BINDING_YAML="tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml"
    export HOSTS="g05glx01,g05glx02"
    export RANKFILE=/etc/mpirun/rankfile_g05glx01_g05glx02
    export TCP_INTERFACE="cnx1"
    mkdir -p logs
    mkdir -p generated/artifacts

    if ! test -f "$RANKFILE"; then
        echo "File '$RANKFILE' does not exist."
        exit 1
    fi
    if ! test -f "$RANK_BINDING_YAML"; then
        echo "File '$RANK_BINDING_YAML' does not exist."
        exit 1
    fi

    export DEEPSEEK_V3_HF_MODEL="/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528"
    export DEEPSEEK_V3_CACHE="/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI"
    export MESH_DEVICE="DUAL"
}

# Run deepseek v3 module tests (models/demos/deepseek_v3/tests)
run_quad_galaxy_deepseekv3_module_tests() {
    fail=0
    setup_dual_galaxy_env

    local MPI_ARGS="--host $HOSTS --map-by rankfile:file=$RANKFILE --bind-to none --output-filename logs/mpi_job"
    # Note: venv activation not needed here - tt-run passes VIRTUAL_ENV and PATH from the calling shell
    # DEEPSEEK_ and MESH_ env vars are passed through by tt-run
    tt-run --tcp-interface "$TCP_INTERFACE" --rank-binding "$RANK_BINDING_YAML" \
        --mpi-args "$MPI_ARGS" \
        pytest -svvv models/demos/deepseek_v3/tests ; fail+=$?

    if [[ $fail -ne 0 ]]; then
        exit 1
    fi
}

# Run teacher forced accuracy test and save metrics to artifacts
run_quad_galaxy_teacher_forced_test() {
    fail=0
    setup_dual_galaxy_env

    local MPI_ARGS="--host $HOSTS --map-by rankfile:file=$RANKFILE --bind-to none --output-filename logs/mpi_job"
    # Note: venv activation not needed here - tt-run passes VIRTUAL_ENV and PATH from the calling shell
    # DEEPSEEK_ and MESH_ env vars are passed through by tt-run
    tt-run --tcp-interface "$TCP_INTERFACE" --rank-binding "$RANK_BINDING_YAML" \
        --mpi-args "$MPI_ARGS" \
        bash -c "pytest -svvv models/demos/deepseek_v3/demo/test_demo_teacher_forced.py::test_demo_teacher_forcing_accuracy 2>&1 | tee generated/artifacts/teacher_forced_output.log" ; fail+=$?

    # Extract accuracy metrics from logs and save to artifact file
    if [[ -f generated/artifacts/teacher_forced_output.log ]]; then
        echo "Extracting accuracy metrics from test output..."
        grep -E "Top-1 accuracy:|Top-5 accuracy:" generated/artifacts/teacher_forced_output.log > generated/artifacts/teacher_forced_accuracy.txt || true
        echo "Accuracy metrics saved to generated/artifacts/teacher_forced_accuracy.txt"
    fi

    if [[ $fail -ne 0 ]]; then
        exit 1
    fi
}

# Run dual demo test (256 prompts, 1 batch) - full_demo variant
run_quad_galaxy_dual_demo_test() {
    fail=0
    setup_dual_galaxy_env

    local MPI_ARGS="--host $HOSTS --map-by rankfile:file=$RANKFILE --bind-to none --output-filename logs/mpi_job"
    # Note: venv activation not needed here - tt-run passes VIRTUAL_ENV and PATH from the calling shell
    # DEEPSEEK_ and MESH_ env vars are passed through by tt-run
    tt-run --tcp-interface "$TCP_INTERFACE" --rank-binding "$RANK_BINDING_YAML" \
        --mpi-args "$MPI_ARGS" \
        bash -c "pytest -svvv 'models/demos/deepseek_v3/demo/test_demo_dual.py::test_demo_dual[full_demo]' 2>&1 | tee generated/artifacts/dual_demo_output.log" ; fail+=$?

    if [[ $fail -ne 0 ]]; then
        exit 1
    fi
}

# Run stress dual demo test (56 prompts, 20 batches) - stress_demo variant
run_quad_galaxy_dual_demo_stress_test() {
    fail=0
    setup_dual_galaxy_env

    local MPI_ARGS="--host $HOSTS --map-by rankfile:file=$RANKFILE --bind-to none --output-filename logs/mpi_job"
    # Note: venv activation not needed here - tt-run passes VIRTUAL_ENV and PATH from the calling shell
    # DEEPSEEK_ and MESH_ env vars are passed through by tt-run
    tt-run --tcp-interface "$TCP_INTERFACE" --rank-binding "$RANK_BINDING_YAML" \
        --mpi-args "$MPI_ARGS" \
        bash -c "pytest -svvv 'models/demos/deepseek_v3/demo/test_demo_dual.py::test_demo_dual[stress_demo]' 2>&1 | tee generated/artifacts/dual_demo_stress_output.log" ; fail+=$?

    if [[ $fail -ne 0 ]]; then
        exit 1
    fi
}

# Legacy function that runs all dual galaxy tests on quad galaxy
run_dual_galaxy_deepseekv3_tests_on_quad_galaxy() {
    run_quad_galaxy_deepseekv3_module_tests
    run_quad_galaxy_teacher_forced_test
    run_quad_galaxy_dual_demo_test
    run_quad_galaxy_dual_demo_stress_test
}

run_quad_galaxy_deepseekv3_unit_tests() {
    fail=0

    local RANK_BINDING_YAML="tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml"
    local MPI_ARGS_BASE="--map-by rankfile:file=/etc/mpirun/rankfile"
    local MPI_ARGS="--host g05glx04,g05glx03,g05glx02,g05glx01 ${MPI_ARGS_BASE}"
    local TCP_INTERFACE="cnx1"

    export DEEPSEEK_V3_HF_MODEL="/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528"
    export DEEPSEEK_V3_CACHE="/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI"
    export MESH_DEVICE="QUAD"

    # Note: venv activation not needed here - tt-run passes VIRTUAL_ENV and PATH from the calling shell
    # DEEPSEEK_ and MESH_ env vars are passed through by tt-run
    tt-run --tcp-interface $TCP_INTERFACE --rank-binding "$RANK_BINDING_YAML" \
        --mpi-args "$MPI_ARGS" \
        pytest -svvv models/demos/deepseek_v3/tests/unit ; fail+=$?

    if [[ $fail -ne 0 ]]; then
        exit 1
    fi
}

run_quad_galaxy_tests() {
  run_quad_galaxy_unit_tests
  run_quad_galaxy_deepseekv3_unit_tests
  run_dual_galaxy_deepseekv3_tests_on_quad_galaxy
}

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

  # Run tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  # Support running specific test function via argument
  local test_function="${1:-all}"

  case "$test_function" in
    "unit_tests")
      run_quad_galaxy_unit_tests
      ;;
    "deepseekv3_unit_tests")
      run_quad_galaxy_deepseekv3_unit_tests
      ;;
    "deepseekv3_module_tests")
      run_quad_galaxy_deepseekv3_module_tests
      ;;
    "teacher_forced")
      run_quad_galaxy_teacher_forced_test
      ;;
    "dual_demo")
      run_quad_galaxy_dual_demo_test
      ;;
    "dual_demo_stress")
      run_quad_galaxy_dual_demo_stress_test
      ;;
    "deepseekv3_integration_tests")
      run_dual_galaxy_deepseekv3_tests_on_quad_galaxy
      ;;
    "all")
      run_quad_galaxy_tests
      ;;
    *)
      echo "Unknown test function: $test_function" 1>&2
      echo "Available options: unit_tests, deepseekv3_unit_tests, deepseekv3_module_tests, teacher_forced, dual_demo, dual_demo_stress, deepseekv3_integration_tests, all" 1>&2
      exit 1
      ;;
  esac
}

main "$@"
