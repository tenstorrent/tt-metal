#!/bin/bash
set -eo pipefail

# Exit immediately if ARCH_NAME is not set or empty
if [ -z "${ARCH_NAME}" ]; then
  echo "Error: ARCH_NAME is not set. Exiting." >&2
  exit 1
fi

###############################################################################
# Infrastructure unit tests (quad galaxy only)
###############################################################################

run_quad_galaxy_unit_tests() {
  fail=0

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

  tt-run --rank-binding "$rank_binding" --mpi-args "$mpi_args" bash -c "source ./python_env/bin/activate && pytest -svv \"tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x16_quad_galaxy\"" ; fail+=$?

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

###############################################################################
# Environment setup helpers
###############################################################################

setup_dual_galaxy_env() {
    export RANK_BINDING_YAML="tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml"
    export HOSTS="g05glx01,g05glx02"
    export RANKFILE=/etc/mpirun/rankfile_g05glx01_g05glx02
    export MPI_ARGS="--host $HOSTS --map-by rankfile:file=$RANKFILE --mca btl self,tcp --mca btl_tcp_if_include cnx1 --bind-to none --output-filename logs/mpi_job --tag-output"
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

setup_quad_galaxy_env() {
    export RANK_BINDING_YAML="tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml"
    export HOSTS="g05glx04,g05glx03,g05glx02,g05glx01"
    export RANKFILE=/etc/mpirun/rankfile
    export MPI_ARGS="--host $HOSTS --map-by rankfile:file=$RANKFILE --mca btl self,tcp --mca btl_tcp_if_include cnx1 --bind-to none --output-filename logs/mpi_job --tag-output"
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
    export MESH_DEVICE="QUAD"
}

# Helper: run a test command via tt-run using the current environment
_run_deepseekv3_tt() {
    tt-run --rank-binding "$RANK_BINDING_YAML" \
        --mpi-args "$MPI_ARGS" \
        bash -c "export DEEPSEEK_V3_HF_MODEL=$DEEPSEEK_V3_HF_MODEL && export DEEPSEEK_V3_CACHE=$DEEPSEEK_V3_CACHE && export MESH_DEVICE=$MESH_DEVICE && source ./python_env/bin/activate && $1"
}

###############################################################################
# DeepSeek V3 unit tests (models/demos/deepseek_v3/tests/unit)
###############################################################################

run_dual_deepseekv3_unit_tests() {
    fail=0
    setup_dual_galaxy_env

    _run_deepseekv3_tt "pytest -svvv models/demos/deepseek_v3/tests/unit" ; fail+=$?

    if [[ $fail -ne 0 ]]; then
        exit 1
    fi
}

run_quad_deepseekv3_unit_tests() {
    fail=0
    setup_quad_galaxy_env

    _run_deepseekv3_tt "pytest -svvv models/demos/deepseek_v3/tests/unit" ; fail+=$?

    if [[ $fail -ne 0 ]]; then
        exit 1
    fi
}

###############################################################################
# DeepSeek V3 module tests (models/demos/deepseek_v3/tests)
###############################################################################

run_dual_deepseekv3_module_tests() {
    fail=0
    setup_dual_galaxy_env

    _run_deepseekv3_tt "pytest -svvv models/demos/deepseek_v3/tests --ignore=models/demos/deepseek_v3/tests/unit --ignore=models/demos/deepseek_v3/tests/fused_op_unit_tests" ; fail+=$?

    if [[ $fail -ne 0 ]]; then
        exit 1
    fi
}

run_quad_deepseekv3_module_tests() {
    fail=0
    setup_quad_galaxy_env

    _run_deepseekv3_tt "pytest -svvv models/demos/deepseek_v3/tests --ignore=models/demos/deepseek_v3/tests/unit --ignore=models/demos/deepseek_v3/tests/fused_op_unit_tests" ; fail+=$?

    if [[ $fail -ne 0 ]]; then
        exit 1
    fi
}

###############################################################################
# Teacher forced accuracy tests
###############################################################################

run_dual_teacher_forced_test() {
    fail=0
    setup_dual_galaxy_env

    _run_deepseekv3_tt "pytest -svvv models/demos/deepseek_v3/demo/test_demo_teacher_forced.py::test_demo_teacher_forcing_accuracy 2>&1 | tee generated/artifacts/dual_teacher_forced_output.log" ; fail+=$?

    # Extract accuracy metrics from logs and save to artifact file
    if [[ -f generated/artifacts/dual_teacher_forced_output.log ]]; then
        echo "Extracting accuracy metrics from test output..."
        grep -E "Top-1 accuracy:|Top-5 accuracy:" generated/artifacts/dual_teacher_forced_output.log > generated/artifacts/dual_teacher_forced_accuracy.txt || true
        echo "Accuracy metrics saved to generated/artifacts/dual_teacher_forced_accuracy.txt"
    fi

    if [[ $fail -ne 0 ]]; then
        exit 1
    fi
}

run_quad_teacher_forced_test() {
    fail=0
    setup_quad_galaxy_env

    _run_deepseekv3_tt "pytest -svvv models/demos/deepseek_v3/demo/test_demo_teacher_forced.py::test_demo_teacher_forcing_accuracy 2>&1 | tee generated/artifacts/quad_teacher_forced_output.log" ; fail+=$?

    # Extract accuracy metrics from logs and save to artifact file
    if [[ -f generated/artifacts/quad_teacher_forced_output.log ]]; then
        echo "Extracting accuracy metrics from test output..."
        grep -E "Top-1 accuracy:|Top-5 accuracy:" generated/artifacts/quad_teacher_forced_output.log > generated/artifacts/quad_teacher_forced_accuracy.txt || true
        echo "Accuracy metrics saved to generated/artifacts/quad_teacher_forced_accuracy.txt"
    fi

    if [[ $fail -ne 0 ]]; then
        exit 1
    fi
}

###############################################################################
# Demo tests (full)
###############################################################################

run_dual_demo_test() {
    fail=0
    setup_dual_galaxy_env

    _run_deepseekv3_tt "pytest -svvv 'models/demos/deepseek_v3/demo/test_demo.py::test_demo[dual_full_demo]' 2>&1 | tee generated/artifacts/dual_demo_output.log" ; fail+=$?

    if [[ $fail -ne 0 ]]; then
        exit 1
    fi
}

run_quad_demo_test() {
    fail=0
    setup_quad_galaxy_env

    _run_deepseekv3_tt "pytest -svvv 'models/demos/deepseek_v3/demo/test_demo.py::test_demo[quad_full_demo]' 2>&1 | tee generated/artifacts/quad_demo_output.log" ; fail+=$?

    if [[ $fail -ne 0 ]]; then
        exit 1
    fi
}

###############################################################################
# Demo stress tests
###############################################################################

run_dual_demo_stress_test() {
    fail=0
    setup_dual_galaxy_env

    _run_deepseekv3_tt "pytest -svvv 'models/demos/deepseek_v3/demo/test_demo.py::test_demo[dual_stress_demo]' 2>&1 | tee generated/artifacts/dual_demo_stress_output.log" ; fail+=$?

    if [[ $fail -ne 0 ]]; then
        exit 1
    fi
}

run_quad_demo_stress_test() {
    fail=0
    setup_quad_galaxy_env

    _run_deepseekv3_tt "pytest -svvv 'models/demos/deepseek_v3/demo/test_demo.py::test_demo[quad_stress_demo]' 2>&1 | tee generated/artifacts/quad_demo_stress_output.log" ; fail+=$?

    if [[ $fail -ne 0 ]]; then
        exit 1
    fi
}

###############################################################################
# Composite runners
###############################################################################

# All dual galaxy deepseek v3 integration tests
run_dual_deepseekv3_integration_tests() {
    run_dual_deepseekv3_module_tests
    run_dual_teacher_forced_test
    run_dual_demo_test
    run_dual_demo_stress_test
}

# All quad galaxy deepseek v3 integration tests
run_quad_deepseekv3_integration_tests() {
    run_quad_deepseekv3_module_tests
    run_quad_teacher_forced_test
    run_quad_demo_test
    run_quad_demo_stress_test
}

# Run everything
run_quad_galaxy_tests() {
    run_quad_galaxy_unit_tests
    run_dual_deepseekv3_unit_tests
    run_quad_deepseekv3_unit_tests
    run_dual_deepseekv3_integration_tests
    run_quad_deepseekv3_integration_tests
}

###############################################################################
# Main dispatcher
###############################################################################

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
        "dual_deepseekv3_unit_tests")
            run_dual_deepseekv3_unit_tests
            ;;
        "quad_deepseekv3_unit_tests")
            run_quad_deepseekv3_unit_tests
            ;;
        "dual_deepseekv3_module_tests")
            run_dual_deepseekv3_module_tests
            ;;
        "quad_deepseekv3_module_tests")
            run_quad_deepseekv3_module_tests
            ;;
        "dual_teacher_forced")
            run_dual_teacher_forced_test
            ;;
        "quad_teacher_forced")
            run_quad_teacher_forced_test
            ;;
        "dual_demo")
            run_dual_demo_test
            ;;
        "quad_demo")
            run_quad_demo_test
            ;;
        "dual_demo_stress")
            run_dual_demo_stress_test
            ;;
        "quad_demo_stress")
            run_quad_demo_stress_test
            ;;
        "dual_deepseekv3_integration_tests")
            run_dual_deepseekv3_integration_tests
            ;;
        "quad_deepseekv3_integration_tests")
            run_quad_deepseekv3_integration_tests
            ;;
        "all")
            run_quad_galaxy_tests
            ;;
        *)
            echo "Unknown test function: $test_function" 1>&2
            echo "Available options: unit_tests, dual_deepseekv3_unit_tests, quad_deepseekv3_unit_tests, dual_deepseekv3_module_tests, quad_deepseekv3_module_tests, dual_teacher_forced, quad_teacher_forced, dual_demo, quad_demo, dual_demo_stress, quad_demo_stress, dual_deepseekv3_integration_tests, quad_deepseekv3_integration_tests, all" 1>&2
            exit 1
            ;;
    esac
}

main "$@"
