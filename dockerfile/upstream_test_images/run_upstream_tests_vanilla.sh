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

test_suite_bh_single_pcie_llama_demo_tests() {
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

    echo "[upstream-tests] Running BH upstream Llama demo model tests"
    # TODO: remove me , just testing this out
    pip3 install -r models/tt_transformers/requirements.txt
    pytest models/tt_transformers/demo/simple_text_demo.py -k performance-batch-1
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
        echo "Error: Test suite '$test_suite' is not part of the '$hw_topology' hw/topology"
        echo "Available test suites for $hw_topology:"
        echo "${hw_topology_test_suites[$hw_topology]}" | sed 's/^/  - /'
        exit 1
    fi

    $test_suite
else
    # Run all test suites for the specified hardware topology
    while IFS= read -r test_func; do
        [[ -n "$test_func" ]] && $test_func
    done <<< "${hw_topology_test_suites[$hw_topology]}"
fi
