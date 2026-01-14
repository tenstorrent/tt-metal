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
    ./build/test/tt_metal/unit_tests_dispatch --gtest_filter=UnitMeshCQSingleCardProgramFixture.*
    ./build/test/tt_metal/unit_tests_dispatch --gtest_filter=UnitMeshCQProgramFixture.*
    ./build/test/tt_metal/unit_tests_dispatch --gtest_filter=*RandomProgramFixture.*
    ./build/test/tt_metal/unit_tests_dispatch --gtest_filter=UnitMeshCQSingleCardBufferFixture.* # Tests EnqueueRead/EnqueueWrite Buffer from DRAM/L1

    TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_api --gtest_filter=*SimpleDram*:*SimpleL1* # Executable is dependent on arch (provided through GitHub CI workflow scripts)
}

# Function test run BH UMD tests, should be any topology
test_suite_bh_umd_unit_tests() {
    ./build/test/umd/blackhole/unit_tests
    # Filter out the test that is failing due to local YAML files, see: https://github.com/tenstorrent/tt-metal/issues/24359
    gtest_filter="-ApiClusterTest.DifferentConstructors"
    ./build/test/umd/api/api_tests --gtest_filter="$gtest_filter"
}

# Function to run BH single PCIe small ML model tests
test_suite_bh_single_pcie_small_ml_model_tests() {
    echo "[upstream-tests] Running BH upstream small model tests"
    pytest --disable-warnings --input-path="models/demos/whisper/demo/dataset/conditional_generation" models/demos/whisper/demo/demo.py::test_demo_for_conditional_generation
    pytest models/demos/blackhole/resnet50/tests/upstream_pipeline
}

test_suite_bh_pcie_didt_tests() {
    echo "[upstream-tests] Running BH upstream didt tests"
    pytest tests/didt/test_resnet_conv.py::test_resnet_conv -k "all" --didt-workload-iterations 100 --determinism-check-interval 1
    pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul -k "without_gelu and all" --didt-workload-iterations 100 --determinism-check-interval 1
    pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul -k "with_gelu and all" --didt-workload-iterations 100 --determinism-check-interval 1
    pytest tests/didt/test_lm_head_matmul.py::test_lm_head_matmul -k "all" --didt-workload-iterations 100 --determinism-check-interval 1
}

verify_llama_dir_() {
    if [ -z "${LLAMA_DIR:-}" ]; then
      echo "LLAMA_DIR environment variable not set. Checking for HF_MODEL and TT_CACHE_PATH..."

      # Check if both HF_MODEL and TT_CACHE_PATH are set
      if [ -z "${HF_MODEL:-}" ] || [ -z "${TT_CACHE_PATH:-}" ]; then
        echo "Error: HF_MODEL and TT_CACHE_PATH environment variables not detected. Please set these environment variables to tell the tests where to find the downloaded Llama weights." >&2
        exit 1
      fi

      # Check if the HF_MODEL directory exists and is not empty
      if [ -d "$HF_MODEL" ] && [ "$(ls -A $HF_MODEL)" ]; then
        echo "[upstream-tests] Llama weights exist, continuing"
      else
        echo "[upstream-tests] Error: Llama weights do not seem to exist in $HF_MODEL, exiting" >&2
        exit 1
      fi
      echo "[upstream-tests] HF_MODEL and TT_CACHE_PATH are set and exist, continuing"
      return 0
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

    pytest models/tt_transformers/demo/simple_text_demo.py -k performance-batch-1
}

test_suite_bh_multi_pcie_metal_unit_tests() {
    echo "[upstream-tests] Running BH LLMBox metal unit tests"

    # Sim HW deskbox has 8 connections so we need to pass in the min-connections arg
    # This changes the connection count assert == 4 to assert >= 4
    if [[ "$hw_topology" == "blackhole_deskbox" ]]; then
        local min_connections_arg="--min-connections 4"
    else
        local min_connections_arg=""
    fi

    # Health check loop. Needed due to the following issues:
    # https://tenstorrent.atlassian.net/browse/SYS-1634
    # https://tenstorrent.atlassian.net/browse/BH-84
    for i in {1..10}; do
        echo "Health check attempt $i"
        if tt-smi -r >/dev/null 2>&1 && ./build/test/tt_metal/tt_fabric/test_system_health $min_connections_arg; then
            echo "Health checks passed"
            break
        fi
        if [ $i -eq 10 ]; then
            echo "Health checks failed after 10 attempts"
            exit 1
        fi
        echo "Health checks failed, retrying..."
        sleep 5
    done
    ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric1D*Fixture.*"
    ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"

    ./build/test/tt_metal/unit_tests_eth
    if [[ "$hw_topology" == "blackhole_llmbox" ]]; then
        pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/Sys_eng_smoke_tests/test_ccl_smoke_test_qb.py
    elif [[ "$hw_topology" == "blackhole_qb_ge" ]]; then
        pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/Sys_eng_smoke_tests/test_ccl_smoke_test_qb_ge.py
    elif [[ "$hw_topology" == "blackhole_loudbox" ]]; then
        pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/Sys_eng_smoke_tests/test_ccl_smoke_test_lb.py
    elif [[ "$hw_topology" == "blackhole_p300" ]]; then
        pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/Sys_eng_smoke_tests/test_ccl_smoke_test_p300.py
    fi
}

test_suite_bh_multi_pcie_llama_demo_tests() {
    echo "[upstream-tests] Running BH multi-pcie upstream Llama demo model tests for topology: $hw_topology"

    if [[ "$hw_topology" == "blackhole_deskbox" ]] || [[ "$hw_topology" == "blackhole_p300" ]]; then
        local data_parallel_devices="2"
    elif [[ "$hw_topology" == "blackhole_llmbox" ]] || [[ "$hw_topology" == "blackhole_qb_ge" ]]; then
        local data_parallel_devices="4"
    elif [[ "$hw_topology" == "blackhole_loudbox" ]]; then
        local data_parallel_devices="8"
    else
        echo "Your blackhole hw topology is not supported to run Llama demo model tests!"
    fi

    echo "Using data_parallel = $data_parallel_devices for topology: $hw_topology"

    pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and ci-32" --data_parallel "$data_parallel_devices"
    pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-stress-1" --data_parallel "$data_parallel_devices" --max_generated_tokens 220
}

test_suite_bh_multi_pcie_llama_stress_tests() {
    echo "[upstream-tests] Running BH multi-pcie upstream Llama stress model tests for topology: $hw_topology"

    if [[ "$hw_topology" == "blackhole_deskbox" ]] || [[ "$hw_topology" == "blackhole_p300" ]]; then
        local data_parallel_devices="2"
    elif [[ "$hw_topology" == "blackhole_llmbox" ]] || [[ "$hw_topology" == "blackhole_qb_ge" ]]; then
        local data_parallel_devices="4"
    elif [[ "$hw_topology" == "blackhole_loudbox" ]]; then
        local data_parallel_devices="8"
    else
        echo "Your blackhole hw topology is not supported to run Llama demo stress tests!"
    fi

    echo "Using data_parallel = $data_parallel_devices for topology: $hw_topology"

    pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-stress-1" --data_parallel "$data_parallel_devices" --max_generated_tokens 22000
}

test_suite_wh_6u_metal_unit_tests() {
    echo "[upstream-tests] running WH 6U upstream metalium unit tests. Note that skips should be treated as failures"
    TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="UnitMeshCQSingleCardFixture.*"
    TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="UnitMeshCQSingleCardProgramFixture.*"
    TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="UnitMeshCQSingleCardBufferFixture.ShardedBufferLarge*ReadWrites"
    TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
}

test_suite_wh_6u_metal_torus_xy_health_check_tests() {
    echo "[upstream-tests] Checking for XY Torus topology on WH 6U"
    ./build/tools/scaleout/run_cluster_validation --cabling-descriptor-path tt_metal/fabric/cabling_descriptors/wh_galaxy_xy_torus.textproto --hard-fail --send-traffic
}

test_suite_wh_6u_metal_qsfp_links_health_check_tests() {
    echo "[upstream-tests] Checking QSFP links on WH 6U (Only works on XY (2D) Torus systems. Check https://github.com/tenstorrent/tt-metal/issues/30415 for updates)"
    ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config ${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_deadlock_stability_6U_galaxy.yaml
}

test_suite_wh_6u_model_unit_tests() {
    echo "[upstream-tests] running WH 6U upstream model unit tests"
    pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py
    pytest tests/ttnn/unit_tests/operations/transformers/test_prefetcher_TG.py
    pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_matmul_1d_gather_in0.py::test_matmul_1d_ring_llama_perf
}

test_suite_wh_6u_llama_demo_tests() {
    echo "[upstream-tests] running WH 6U upstream Llama demo tests with weights"

    verify_llama_dir_

    pytest models/demos/llama3_70b_galaxy/tests/test_llama_model.py -k "quick"
    pytest models/demos/llama3_70b_galaxy/tests/unit_tests/test_llama_model_prefill.py
    pytest models/demos/llama3_70b_galaxy/demo/text_demo.py -k "repeat"
    # Some AssertionError: Throughput is out of targets 49 - 53 t/s/u in 200 iterations
    # assert 200 <= 20
    # pytest models/demos/llama3_70b_galaxy/demo/demo_decode.py -k "full"
    pytest models/demos/llama3_70b_galaxy/demo/demo_decode.py -k "mini-stress-test"
}

test_suite_wh_6u_llama_long_stress_tests() {
    echo "[upstream-tests] running WH 6U upstream Llama long stress tests. Note that on 6U systems built as of End of May 2025, this may take up to 4 hours to run."
    echo "[upstream-tests] Ensure that you have a TG directory populated with .bin files in LLAMA_DIR on the host."

    verify_llama_dir_

    # This will take almost 3 hours. Ensure that the tensors are cached in the LLAMA_DIR.
    pytest models/demos/llama3_70b_galaxy/demo/demo_decode.py -k "stress-test and not mini-stress-test"
}

test_suite_bh_ttnn_stress_tests() {
    echo "[upstream-tests] running BH upstream ttnn stress tests"
    pytest tests/ttnn/stress_tests/
}

test_suite_bh_glx_metal_unit_tests() {
    echo "[upstream-tests] running BH GLX upstream metal unit tests"

    # BH Galaxy XY (2D) Torus System Validation (no fabric, simply validate that expected links are discovered and healthy)
    ./build/tools/scaleout/run_cluster_validation --cabling-descriptor-path tools/tests/scaleout/cabling_descriptors/bh_galaxy_xy_torus.textproto --hard-fail --send-traffic
    RELIABILITY_MODE=relaxed ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*Fabric2D*.*"
    RELIABILITY_MODE=relaxed ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*Fabric1D*.*":-NightlyFabric1DFixture.TestEDMConnectionStressTestQuick
    RELIABILITY_MODE=relaxed TT_METAL_CLEAR_L1=1 build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_sanity_common.yaml
    # Deadlock stability tests - These validate 2D Torus (QSFP Link) stability
    RELIABILITY_MODE=relaxed TT_METAL_CLEAR_L1=1 build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_deadlock_stability_bh_6U_galaxy.yaml

    # Dispatch
    build/test/tt_metal/unit_tests_eth --gtest_filter=UnitMeshCQMultiDeviceProgramFixture.ActiveEthKernelsSendInterleavedBufferAllConnectedChips

    build/test/tt_metal/unit_tests_dispatch --gtest_filter="\
UnitMeshCQProgramFixture.TensixTestRandomizedProgram:\
UnitMeshRandomProgramFixture.TensixActiveEthTestPrograms:\
UnitMeshRandomProgramFixture.TensixTestLargeProgramInBetweenFiveSmallPrograms:\
UnitMeshRandomProgramTraceFixture.TensixActiveEthTestProgramsTraceAndNoTrace:\
UnitMeshRandomProgramTraceFixture.TensixActiveEthTestProgramsTrace:\
UnitMeshRandomProgramTraceFixture.TensixTestLargeProgramInBetweenFiveSmallProgramsTrace:\
UnitMeshRandomProgramTraceFixture.TensixTestSimpleProgramsTrace:\
UnitMeshCQTraceFixture.TensixEnqueueMultiProgramTraceBenchmark:\
UnitMeshCQTraceFixture.TensixEnqueueTwoProgramTrace:\
UnitMeshCQSingleCardBufferFixture.ShardedBufferLargeL1ReadWrites:\
UnitMeshCQSingleCardBufferFixture.ShardedBufferLargeDRAMReadWrites:\
UnitMeshCQSingleCardFixture.TensixTestSubDeviceAllocations:\
UnitMeshMultiCQMultiDeviceEventFixture.*:\
UnitMeshCQSingleCardFixture.TensixTestReadWriteMultipleCoresL1"
}

test_suite_bh_glx_python_unit_tests() {
    echo "[upstream-tests] running BH GLX upstream python unit tests"
    # CCL / Ops
    pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/Sys_eng_smoke_tests/test_ccl_smoke_test_galaxy_torus.py
}

test_suite_bh_glx_llama_demo_tests() {
    echo "[upstream-tests] running BH GLX upstream Llama demo tests with weights"

    verify_llama_dir_

    pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and ci-32" --data_parallel 32
}

test_suite_bh_glx_torus_xyz_health_check_tests() {
    echo "[upstream-tests] Checking for XY Torus + Z links topology on BH 6U Galaxy"
    # Fabric
    # This test is to be run on systems that have the XY Torus links setup, along with Z connections between adjacent trays.
    # The purpose of this test is to verify that the Z Ports are healthy, and is to be run by operators/technicians installing BH Galaxies.
    # This test is not to be run on officical topologies (Mesh, X Torus, Y Torus or XY Torus).
    ./build/tools/scaleout/run_cluster_validation --cabling-descriptor-path tools/tests/scaleout/cabling_descriptors/bh_galaxy_xy_torus_z_ports.textproto --hard-fail --send-traffic
}

# Define test suite mappings for different hardware topologies
declare -A hw_topology_test_suites

# Store test suites as newline-separated lists
hw_topology_test_suites["blackhole"]="
test_suite_bh_umd_unit_tests
test_suite_bh_pcie_didt_tests
test_suite_bh_single_pcie_python_unit_tests
test_suite_bh_single_pcie_metal_unit_tests
test_suite_bh_single_pcie_small_ml_model_tests
test_suite_bh_single_pcie_llama_demo_tests" # NOTE: This test MUST be last because of the requirements install currently in the llama tests

hw_topology_test_suites["blackhole_no_models"]="
test_suite_bh_umd_unit_tests
test_suite_bh_pcie_didt_tests
test_suite_bh_single_pcie_python_unit_tests
test_suite_bh_single_pcie_metal_unit_tests"

hw_topology_test_suites["blackhole_llmbox"]="
test_suite_bh_multi_pcie_metal_unit_tests
test_suite_bh_pcie_didt_tests
test_suite_bh_multi_pcie_llama_demo_tests"

hw_topology_test_suites["blackhole_deskbox"]="
test_suite_bh_multi_pcie_metal_unit_tests
test_suite_bh_pcie_didt_tests
test_suite_bh_multi_pcie_llama_demo_tests"

hw_topology_test_suites["blackhole_loudbox"]="
test_suite_bh_multi_pcie_metal_unit_tests
test_suite_bh_pcie_didt_tests
test_suite_bh_multi_pcie_llama_demo_tests"

hw_topology_test_suites["blackhole_p300"]="
test_suite_bh_umd_unit_tests
test_suite_bh_single_pcie_metal_unit_tests
test_suite_bh_multi_pcie_metal_unit_tests
test_suite_bh_pcie_didt_tests
test_suite_bh_multi_pcie_llama_demo_tests"

hw_topology_test_suites["blackhole_qb_ge"]="
test_suite_bh_multi_pcie_metal_unit_tests
test_suite_bh_pcie_didt_tests
test_suite_bh_multi_pcie_llama_demo_tests"

# test_suite_wh_6u_llama_demo_tests was removed because of
# https://github.com/tenstorrent/tt-metal/issues/34990
hw_topology_test_suites["wh_6u"]="
test_suite_wh_6u_metal_torus_xy_health_check_tests
test_suite_wh_6u_model_unit_tests
test_suite_wh_6u_metal_unit_tests
test_suite_wh_6u_metal_qsfp_links_health_check_tests"

hw_topology_test_suites["blackhole_ttnn_stress_tests"]="
test_suite_bh_ttnn_stress_tests"

hw_topology_test_suites["blackhole_glx"]="
test_suite_bh_glx_metal_unit_tests
test_suite_bh_glx_python_unit_tests
test_suite_bh_glx_llama_demo_tests"

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
