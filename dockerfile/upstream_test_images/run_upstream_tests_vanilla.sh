#!/usr/bin/env bash
set -euo pipefail

# Function to run BH single PCIe Python unit tests
test_suite_bh_single_pcie_python_unit_tests() {
    echo "[upstream-tests] Run BH python upstream tests"
    pytest --collect-only tests/ttnn/unit_tests
}

# Function to run BH single PCIe metal unit tests
test_suite_bh_single_pcie_metal_unit_tests() {
    echo "[upstream-tests] Running BH upstream metal runtime tests"
    ARCH_NAME=blackhole TT_METAL_SLOW_DISPATCH_MODE=1 ./tests/scripts/run_cpp_fd2_tests.sh
    # I wonder why we can't put these in the validation suite?
    ./build/test/tt_metal/unit_tests_dispatch --gtest_filter=CommandQueueSingleCardProgramFixture.*
    ./build/test/tt_metal/unit_tests_dispatch --gtest_filter=CommandQueueProgramFixture.*
    ./build/test/tt_metal/unit_tests_dispatch --gtest_filter=RandomProgramFixture.*
    ./build/test/tt_metal/unit_tests_dispatch --gtest_filter=CommandQueueSingleCardBufferFixture.* # Tests EnqueueRead/EnqueueWrite Buffer from DRAM/L1
    TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_api --gtest_filter=*SimpleDram*:*SimpleL1* # Executable is dependent on arch (provided through GitHub CI workflow scripts)
}

# Function to run BH single PCIe small ML model tests
test_suite_bh_single_pcie_small_ml_model_tests() {
    echo "[upstream-tests] Running BH upstream small model tests"
    pytest --disable-warnings --input-path="models/demos/whisper/demo/dataset/conditional_generation" models/demos/whisper/demo/demo.py::test_demo_for_conditional_generation
    pytest models/demos/blackhole/resnet50/tests/upstream_pipeline
}

verify_llama_dir_() {
    if [ -z "${LLAMA_DIR}" ]; then
      echo "Error: LLAMA_DIR environment variable not detected. Please set this environment variable to tell the tests where to find the downloaded Llama weights." >&2
      exit 1
    fi

    if [ -d "$LLAMA_DIR" ] && [ "$(ls -A $LLAMA_DIR)" ]; then
      echo "[upstream-tests] Llama weights exist, continuing"
    else
      echo "[upstream-tests] Error: Llama weights do not seem to exist in $LLAMA_DIR, exiting" >&2
      exit 1
    fi
}

test_suite_bh_single_pcie_llama_demo_tests() {
    echo "[upstream-tests] Running BH upstream Llama demo model tests"

    verify_llama_dir_

    # TODO: remove me , just testing this out
    pip3 install -r models/tt_transformers/requirements.txt
    pytest models/tt_transformers/demo/simple_text_demo.py -k performance-batch-1
}

test_suite_bh_single_pcie_llama_demo_tests() {
    echo "[upstream-tests] Running BH upstream Llama demo model tests"

    verify_llama_dir_

    # TODO: remove me , just testing this out
    pip3 install -r models/tt_transformers/requirements.txt
    pytest models/tt_transformers/demo/simple_text_demo.py -k performance-batch-1
}

test_suite_bh_llmbox_llama_demo_tests() {
    echo "[upstream-tests] Running BH LLMBox upstream Llama demo model tests"

    verify_llama_dir_

    # TODO: remove me once upgraded
    pip3 install -r models/tt_transformers/requirements.txt
    pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and ci-32" --data_parallel 4
}

test_suite_wh_6u_metal_unit_tests() {
    echo "[upstream-tests] running WH 6U upstream metalium unit tests. Note that skips should be treated as failures"
    ./build/test/tt_metal/tt_fabric/test_system_health
    TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCardFixture.*"
    TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCardProgramFixture.*"
    TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCardBufferFixture.ShardedBufferLarge*ReadWrites"
    TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
}

test_suite_wh_6u_metal_2d_torus_health_check_tests() {
    echo "[upstream-tests] Checking for 2D Torus topology on WH 6U"
    ./build/test/tt_metal/tt_fabric/test_system_health --system-topology TORUS_2D
}

test_suite_wh_6u_model_unit_tests() {
    echo "[upstream-tests] running WH 6U upstream model unit tests"
    pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py
    pytest tests/ttnn/unit_tests/operations/test_prefetcher_TG.py
    pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_matmul_1d_gather_in0.py::test_matmul_1d_ring_llama_perf
    pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py
    # pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py hang???
}

test_suite_wh_6u_llama_demo_tests() {
    echo "[upstream-tests] running WH 6U upstream Llama demo tests with weights"

    verify_llama_dir_

    # TODO: to remove...
    pip install -r models/tt_transformers/requirements.txt
    pytest models/demos/llama3_subdevices/tests/test_llama_model.py -k "quick"
    pytest models/demos/llama3_subdevices/tests/unit_tests/test_llama_model_prefill.py
    pytest models/demos/llama3_subdevices/demo/text_demo.py -k "repeat"
    # Some AssertionError: Throughput is out of targets 49 - 53 t/s/u in 200 iterations
    # assert 200 <= 20
    # pytest models/demos/llama3_subdevices/demo/demo_decode.py -k "full"
    pytest models/demos/llama3_subdevices/demo/demo_decode.py -k "mini-stress-test"
}

test_suite_wh_6u_llama_long_stress_tests() {
    echo "[upstream-tests] running WH 6U upstream Llama long stress tests. Note that on 6U systems built as of End of May 2025, this may take up to 4 hours to run."
    echo "[upstream-tests] Ensure that you have a TG directory populated with .bin files in LLAMA_DIR on the host."

    verify_llama_dir_

    # TODO: to remove...
    pip install -r models/tt_transformers/requirements.txt
    # This will take almost 3 hours. Ensure that the tensors are cached in the LLAMA_DIR.
    pytest models/demos/llama3_subdevices/demo/demo_decode.py -k "stress-test and not mini-stress-test"
}

# Define test suite mappings for different hardware topologies
declare -A hw_topology_test_suites

# Store test suites as newline-separated lists
hw_topology_test_suites["blackhole"]="test_suite_bh_single_pcie_python_unit_tests
test_suite_bh_single_pcie_metal_unit_tests
test_suite_bh_single_pcie_small_ml_model_tests
test_suite_bh_single_pcie_llama_demo_tests" # NOTE: This test MUST be last because of the requirements install currently in the llama tests

hw_topology_test_suites["blackhole_no_models"]="test_suite_bh_single_pcie_python_unit_tests
test_suite_bh_single_pcie_metal_unit_tests"

hw_topology_test_suites["blackhole_llmbox"]="test_suite_bh_llmbox_llama_demo_tests"

hw_topology_test_suites["wh_6u"]="test_suite_wh_6u_model_unit_tests
test_suite_wh_6u_llama_demo_tests
test_suite_wh_6u_metal_unit_tests
test_suite_wh_6u_metal_2d_torus_health_check_tests"

# Function to display help
show_help() {
    echo "Usage: $0 [options] <hw_topology>"
    echo "  hw_topology    Required. The hardware/topology to run tests against."
    echo "  -h, --help     Show this help message."
    echo "  --test-suite   Optional. Specify which test suite to run. Available options:"
    # Dynamically list available test functions
    declare -F | grep "test_suite_" | awk '{print "                 - " $3}'
    echo "                 If not specified, all test suites will run."
}

# Initialize variables
test_suite=""
hw_topology=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        --test-suite)
            if [[ -z "$2" ]]; then
                echo "Error: --test-suite requires a value"
                show_help
                exit 1
            fi
            test_suite="$2"
            shift 2
            ;;
        *)
            if [[ -z "$hw_topology" ]]; then
                hw_topology="$1"
            else
                echo "Error: Unexpected argument: $1"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate hw_topology is provided
if [[ -z "$hw_topology" ]]; then
    echo "Error: hw_topology argument is required. You need to specify what kind of hardware topology you want to run the tests on."
    show_help
    exit 1
fi

# Check if the test suite is part of the specified hardware topology
if [[ -z "${hw_topology_test_suites[$hw_topology]:-}" ]]; then
    echo "Error: Unsupported hw/topology: $hw_topology"
    echo "We support the following: ${!hw_topology_test_suites[@]}"
    exit 1
fi

# Validate test_suite if provided
if [[ -n "$test_suite" ]]; then
    # Check if the function exists
    if ! declare -F "$test_suite" > /dev/null; then
        echo "Error: The requested test suite $test_suite is not available. Please check the help for available test suites."
        show_help
        exit 1
    fi

    # Check if the test suite is in the list of test suites for this topology
    if ! echo "${hw_topology_test_suites[$hw_topology]}" | grep -q "^$test_suite$"; then
        echo "[upstream-tests] Warning: Test suite '$test_suite' is not part of the '$hw_topology' hw/topology"
        echo "Available test suites for $hw_topology:"
        echo "${hw_topology_test_suites[$hw_topology]}" | sed 's/^/  - /'
    fi

    $test_suite
else
    # Run all test suites for the specified hardware topology
    while IFS= read -r test_func; do
        [[ -n "$test_func" ]] && $test_func
    done <<< "${hw_topology_test_suites[$hw_topology]}"
fi
