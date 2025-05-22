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
}

# Define test suite mappings for different hardware topologies
declare -A hw_topology_test_suites

hw_topology_test_suites["blackhole"]="test_suite_bh_single_pcie_python_unit_tests test_suite_bh_single_pcie_metal_unit_tests test_suite_bh_single_pcie_small_ml_model_tests"
hw_topology_test_suites["blackhole_no_models"]="test_suite_bh_single_pcie_python_unit_tests test_suite_bh_single_pcie_metal_unit_tests"

# Function to display help
show_help() {
    echo "Usage: $0 [options] <hw_topology>"
    echo "  hw_topology    Required. The hardware topology to run tests against."
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

# Validate test_suite if provided
if [[ -n "$test_suite" ]]; then
    # Check if the function exists
    if ! declare -F "$test_suite" > /dev/null; then
        echo "Error: Invalid test suite: $test_suite"
        show_help
        exit 1
    fi
fi

if [[ -n "$test_suite" ]]; then
    $test_suite
else
    # Check if the hardware topology exists in our mapping
    if [[ -z "${hw_topology_test_suites[$hw_topology]:-}" ]]; then
        echo "Error: Unsupported hw_topology: $hw_topology"
        echo "We support the following: ${!hw_topology_test_suites[@]}"
        exit 1
    fi

    # Run all test suites for the specified hardware topology
    for test_func in ${hw_topology_test_suites[$hw_topology]}; do
        $test_func
    done
fi
