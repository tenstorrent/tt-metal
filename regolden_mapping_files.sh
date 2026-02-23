#!/bin/bash
set -e

# Script to regenerate all golden mapping files
# This script runs all tests that generate golden mapping files and copies the generated files

BUILD_DIR="build_Debug"
TEST_BINARY="${BUILD_DIR}/test/tt_metal/tt_fabric/fabric_unit_tests"
GOLDEN_DIR="tests/tt_metal/tt_fabric/golden_mapping_files"
GENERATED_DIR="generated/fabric"
TT_RUN="./python_env/bin/tt-run"

# Source python environment for tt-run
if [ -f "./python_env/bin/activate" ]; then
    source ./python_env/bin/activate
fi

# Check if tt-run exists
if [ ! -f "${TT_RUN}" ] && ! command -v tt-run &> /dev/null; then
    echo "Error: tt-run not found at ${TT_RUN} and not in PATH"
    echo "Please ensure python_env is set up and activated"
    exit 1
fi

# Use tt-run from PATH if available, otherwise use the local one
if command -v tt-run &> /dev/null; then
    TT_RUN="tt-run"
fi

# Ensure directories exist
mkdir -p "${GOLDEN_DIR}"
mkdir -p "${GENERATED_DIR}"

echo "=== Regenerating Golden Mapping Files ==="
echo ""

# Function to run single-host test and copy golden file
run_single_host_test() {
    local test_filter=$1
    local cluster_desc=$2
    local mesh_graph_desc=$3
    local golden_file=$4

    echo "Running single-host test: ${test_filter}"

    # Clean up any existing mapping file to ensure we get the one created by this test
    rm -f "${GENERATED_DIR}/asic_to_fabric_node_mapping_rank_1_of_1.yaml"

    local test_output=$(mktemp)
    local test_exit_code=0

    if [ -n "${mesh_graph_desc}" ]; then
        TT_METAL_SLOW_DISPATCH_MODE=1 \
        TT_METAL_MOCK_CLUSTER_DESC_PATH="${cluster_desc}" \
        TT_MESH_GRAPH_DESC_PATH="${mesh_graph_desc}" \
        "${TEST_BINARY}" --gtest_filter="${test_filter}" > "${test_output}" 2>&1 || test_exit_code=$?
    else
        TT_METAL_SLOW_DISPATCH_MODE=1 \
        TT_METAL_MOCK_CLUSTER_DESC_PATH="${cluster_desc}" \
        "${TEST_BINARY}" --gtest_filter="${test_filter}" > "${test_output}" 2>&1 || test_exit_code=$?
    fi

    if [ -f "${GENERATED_DIR}/asic_to_fabric_node_mapping_rank_1_of_1.yaml" ]; then
        cp "${GENERATED_DIR}/asic_to_fabric_node_mapping_rank_1_of_1.yaml" \
           "${GOLDEN_DIR}/${golden_file}"
        echo "  ✓ Copied to ${golden_file}"
        rm -f "${test_output}"
    else
        echo "  ✗ Failed: Generated file not found"
        echo "    Test exit code: ${test_exit_code}"
        echo "    Last 10 lines of test output:"
        tail -10 "${test_output}" | sed 's/^/      /'
        rm -f "${test_output}"
    fi
}

# Function to run multi-host test and copy golden file
run_multi_host_test() {
    local test_filter=$1
    local cluster_mapping=$2
    local rank_binding=$3
    local world_size=$4
    local golden_file=$5
    local extra_mpi_args=${6:-""}

    echo "Running multi-host test: ${test_filter} (world_size=${world_size})"

    # Clean up any existing mapping files for this world_size to ensure we get the one created by this test
    rm -f "${GENERATED_DIR}/asic_to_fabric_node_mapping_rank_"*"_of_${world_size}.yaml"

    local test_output=$(mktemp)
    local test_exit_code=0

    local mpi_args="--allow-run-as-root"
    if [ -n "${extra_mpi_args}" ]; then
        mpi_args="${mpi_args} ${extra_mpi_args}"
    fi

    # Run with timeout to prevent hanging (5 minutes should be enough for most tests)
    timeout 300 "${TT_RUN}" --mock-cluster-rank-binding "${cluster_mapping}" \
           --rank-binding "${rank_binding}" \
           --mpi-args "${mpi_args}" \
           "${TEST_BINARY}" \
           --gtest_filter="${test_filter}" > "${test_output}" 2>&1 || test_exit_code=$?

    # Check if timeout occurred
    if [ ${test_exit_code} -eq 124 ]; then
        echo "  ✗ Failed: Test timed out after 5 minutes"
        echo "    Last 20 lines of test output:"
        tail -20 "${test_output}" | sed 's/^/      /'
        rm -f "${test_output}"
        return
    fi

    local generated_file="${GENERATED_DIR}/asic_to_fabric_node_mapping_rank_1_of_${world_size}.yaml"
    if [ -f "${generated_file}" ]; then
        cp "${generated_file}" "${GOLDEN_DIR}/${golden_file}"
        echo "  ✓ Copied to ${golden_file}"
        rm -f "${test_output}"
    else
        echo "  ✗ Failed: Generated file not found: ${generated_file}"
        echo "    Test exit code: ${test_exit_code}"
        echo "    Searching for test failure details in output..."
        # Show more context around failures
        if grep -q "FAILED" "${test_output}"; then
            echo "    Test failure details:"
            grep -A 10 "FAILED" "${test_output}" | head -20 | sed 's/^/      /'
        fi
        echo "    Last 20 lines of test output:"
        tail -20 "${test_output}" | sed 's/^/      /'
        rm -f "${test_output}"
    fi
}

# Single-host tests
echo "--- Single-Host Tests ---"

# ControlPlaneFixture_SingleGalaxy
run_single_host_test \
    "ControlPlaneFixture.TestSingleGalaxyControlPlaneInit" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml" \
    "" \
    "ControlPlaneFixture_SingleGalaxy.yaml"

# ControlPlaneFixture_SingleGalaxy_1x32
run_single_host_test \
    "ControlPlaneFixture.TestSingleGalaxy1x32ControlPlaneInit" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml" \
    "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_1x32_mesh_graph_descriptor.textproto" \
    "ControlPlaneFixture_SingleGalaxy_1x32.yaml"

# ControlPlaneFixture_T3k
run_single_host_test \
    "ControlPlaneFixture.TestT3kControlPlaneInit" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml" \
    "" \
    "ControlPlaneFixture_T3k.yaml"

# ControlPlaneFixture_Custom2x2
run_single_host_test \
    "ControlPlaneFixture.TestCustom2x2ControlPlaneInit" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/2x2_n300_cluster_desc.yaml" \
    "" \
    "ControlPlaneFixture_Custom2x2.yaml"

# TestControlPlaneInitNoMGD tests (various cluster descriptors)
# 2xp150_disconnected
run_single_host_test \
    "ControlPlaneFixture.TestControlPlaneInitNoMGD" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/2xp150_disconnected_cluster_desc.yaml" \
    "" \
    "ControlPlaneFixture_TestControlPlaneInitNoMGD_2xp150_cluster_desc.yaml"

# 4xn300_disconnected
run_single_host_test \
    "ControlPlaneFixture.TestControlPlaneInitNoMGD" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/4xn300_disconnected_cluster_desc.yaml" \
    "" \
    "ControlPlaneFixture_TestControlPlaneInitNoMGD_4xn300_cluster_desc.yaml"

# bh_galaxy_xyz (no mesh graph)
run_single_host_test \
    "ControlPlaneFixture.TestControlPlaneInitNoMGD" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_galaxy_xyz_cluster_desc.yaml" \
    "" \
    "ControlPlaneFixture_TestControlPlaneInitNoMGD_bh_galaxy_xyz_cluster_desc.yaml"

# bh_galaxy_xyz with single_bh_galaxy mesh graph
run_single_host_test \
    "ControlPlaneFixture.TestControlPlaneInitNoMGD" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_galaxy_xyz_cluster_desc.yaml" \
    "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto" \
    "ControlPlaneFixture_TestControlPlaneInitNoMGD_bh_galaxy_xyz_single_bh_galaxy_cluster_desc.yaml"

# bh_galaxy_xyz with single_bh_galaxy_torus_xy mesh graph
run_single_host_test \
    "ControlPlaneFixture.TestControlPlaneInitNoMGD" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_galaxy_xyz_cluster_desc.yaml" \
    "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto" \
    "ControlPlaneFixture_TestControlPlaneInitNoMGD_bh_galaxy_xyz_single_bh_galaxy_torus_xy_cluster_desc.yaml"

# T3K2x2AssignZDirectionControlPlaneInit - This is actually a MultiHost test, skip here
# It will be handled in the multi-host section

# T3kCustomMeshGraph tests - these are parameterized tests
# We need to run them individually based on the test parameters
echo ""
echo "Running T3kCustomMeshGraph tests..."
TT_METAL_SLOW_DISPATCH_MODE=1 \
TT_METAL_MOCK_CLUSTER_DESC_PATH="tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml" \
"${TEST_BINARY}" --gtest_filter="T3kCustomMeshGraphControlPlaneFixture.*" > /dev/null 2>&1 || true

# Copy T3kCustomMeshGraph golden files if they exist
for file in "${GENERATED_DIR}"/asic_to_fabric_node_mapping_rank_1_of_1.yaml; do
    if [ -f "${file}" ]; then
        # The test generates files with specific names based on parameters
        # We'll need to check what was actually generated
        echo "  Note: T3kCustomMeshGraph tests generate files with parameter-specific names"
    fi
done

# Multi-host tests
echo ""
echo "--- Multi-Host Tests ---"

# TestDualGalaxyControlPlaneInit
run_multi_host_test \
    "MultiHost.TestDualGalaxyControlPlaneInit" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_dual_host_cluster_desc_mapping.yaml" \
    "tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml" \
    2 \
    "TestDualGalaxyControlPlaneInit.yaml"

# TestDualGalaxyControlPlaneInitFlipped
run_multi_host_test \
    "MultiHost.TestDualGalaxyControlPlaneInitFlipped" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/flipped_6u_dual_host_cluster_desc_mapping.yaml" \
    "tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml" \
    2 \
    "TestDualGalaxyControlPlaneInitFlipped.yaml"

# Test6uSplit8x2ControlPlaneInit
run_multi_host_test \
    "MultiHost.Test6uSplit8x2ControlPlaneInit" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_split_8x2_cluster_desc_mapping.yaml" \
    "tests/tt_metal/distributed/config/dual_8x2_rank_bindings.yaml" \
    2 \
    "Test6uSplit8x2ControlPlaneInit.yaml"

# Test6uSplit4x4ControlPlaneInit
run_multi_host_test \
    "MultiHost.Test6uSplit4x4ControlPlaneInit" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_split_4x4_cluster_desc_mapping.yaml" \
    "tests/tt_metal/distributed/config/dual_4x4_rank_bindings.yaml" \
    2 \
    "Test6uSplit4x4ControlPlaneInit.yaml"

# TestDual2x4ControlPlaneInit
run_multi_host_test \
    "MultiHost.TestDual2x4ControlPlaneInit" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_dual_host_cluster_desc_mapping.yaml" \
    "tests/tt_metal/distributed/config/dual_t3k_rank_bindings.yaml" \
    2 \
    "TestDual2x4ControlPlaneInit.yaml"

# TestSplit2x2ControlPlaneInit
run_multi_host_test \
    "MultiHost.TestSplit2x2ControlPlaneInit" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x2_cluster_desc_mapping.yaml" \
    "tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml" \
    2 \
    "TestSplit2x2ControlPlaneInit.yaml"

# TestBigMesh2x4ControlPlaneInit
run_multi_host_test \
    "MultiHost.TestBigMesh2x4ControlPlaneInit" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x4_big_mesh_cluster_desc_mapping.yaml" \
    "tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml" \
    4 \
    "TestBigMesh2x4ControlPlaneInit.yaml"

# Test32x4QuadGalaxyControlPlaneInit
run_multi_host_test \
    "MultiHost.Test32x4QuadGalaxyControlPlaneInit" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/32x4_quad_galaxy_cluster_desc_mapping.yaml" \
    "tests/tt_metal/distributed/config/32x4_quad_galaxy_rank_bindings.yaml" \
    4 \
    "Test32x4QuadGalaxyControlPlaneInit.yaml"

# TestQuadGalaxyControlPlaneInit
run_multi_host_test \
    "MultiHost.TestQuadGalaxyControlPlaneInit" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_quad_host_cluster_desc_mapping.yaml" \
    "tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml" \
    4 \
    "TestQuadGalaxyControlPlaneInit.yaml"

# TestBHQB4x4ControlPlaneInit
run_multi_host_test \
    "MultiHost.TestBHQB4x4ControlPlaneInit" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_qb_4x4_cluster_desc_mapping.yaml" \
    "tests/tt_metal/distributed/config/bh_qb_4x4_rank_bindings.yaml" \
    4 \
    "TestBHQB4x4ControlPlaneInit.yaml"

# TestBHQB4x4RelaxedControlPlaneInit
run_multi_host_test \
    "MultiHost.TestBHQB4x4RelaxedControlPlaneInit" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_qb_4x4_cluster_desc_mapping.yaml" \
    "tests/tt_metal/distributed/config/bh_qb_4x4_rank_bindings.yaml" \
    4 \
    "TestBHQB4x4RelaxedControlPlaneInit.yaml"

# TestClosetBox3PodTTSwitchControlPlaneInit
# Note: Using wh_closetbox_superpod_cluster_desc_mapping.yaml (matches workflow)
run_multi_host_test \
    "MultiHost.TestClosetBox3PodTTSwitchControlPlaneInit" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/wh_closetbox_superpod_cluster_desc_mapping.yaml" \
    "tests/tt_metal/distributed/config/wh_closetbox_3pod_ttswitch_rank_bindings.yaml" \
    3 \
    "TestClosetBox3PodTTSwitchControlPlaneInit.yaml"

# TestClosetBox3PodTTSwitchAPIs
run_multi_host_test \
    "MultiHost.TestClosetBox3PodTTSwitchAPIs" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/wh_closetbox_superpod_cluster_desc_mapping.yaml" \
    "tests/tt_metal/distributed/config/wh_closetbox_3pod_ttswitch_rank_bindings.yaml" \
    3 \
    "TestClosetBox3PodTTSwitchAPIs.yaml"

# BHDualGalaxyControlPlaneInit
run_multi_host_test \
    "MultiHost.BHDualGalaxyControlPlaneInit" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/dual_bh_galaxy_experimental_cluster_desc_mapping.yaml" \
    "tests/tt_metal/distributed/config/dual_bh_galaxy_experimental_rank_bindings.yaml" \
    2 \
    "BHDualGalaxyControlPlaneInit.yaml"

# TestBHBlitzPipelineControlPlaneInit
run_multi_host_test \
    "MultiHost.TestBHBlitzPipelineControlPlaneInit" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/bh_blitz_pipeline_cluster_desc_mapping.yaml" \
    "tests/tt_metal/distributed/config/bh_blitz_pipeline_rank_bindings.yaml" \
    2 \
    "TestBHBlitzPipelineControlPlaneInit.yaml" \
    "--oversubscribe"

# TestTriplePod16x8QuadBHGalaxyControlPlaneInit
run_multi_host_test \
    "MultiHost.TestTriplePod16x8QuadBHGalaxyControlPlaneInit" \
    "tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/3_pod_16x8_bh_galaxy_cluster_desc_mapping.yaml" \
    "tests/tt_metal/distributed/config/triple_16x8_quad_bh_galaxy_rank_bindings.yaml" \
    3 \
    "TestTriplePod16x8QuadBHGalaxyControlPlaneInit.yaml"

echo ""
echo "=== Done ==="
echo "Review the changes with: git diff ${GOLDEN_DIR}/"
