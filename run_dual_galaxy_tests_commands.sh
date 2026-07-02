#!/bin/bash
# Commands to run dual galaxy tests using tt-run
# Based on .github/workflows/multi-host-physical.yaml and tests/scripts/multihost/run_dual_galaxy_tests.sh

# Set up variables
RANK_BINDING="tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml"
RANKFILE="$(pwd)/rankfile_dual_galaxy"
# For SLURM sessions with mpirun, try using --rankfile with --host
# --host is needed for TCP BTL to establish connections between hosts
# --bind-to none is important for multi-host communication
# Using exclude instead of include - cnx1 interface doesn't exist on these hosts
MPI_ARGS_BASE="--host wh-glx-a04u02,wh-glx-a05u02 --rankfile $RANKFILE --mca btl self,tcp --mca btl_tcp_if_exclude docker0,lo,tailscale0 --bind-to none --tag-output"
MPI_ARGS_REVERSED="--host wh-glx-a05u02,wh-glx-a04u02 --rankfile $RANKFILE --mca btl self,tcp --mca btl_tcp_if_exclude docker0,lo,tailscale0 --bind-to none --tag-output"

# Ensure we're in TT_METAL_HOME
if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Error: TT_METAL_HOME is not set. Please set it to the root of your tt-metal repository."
    exit 1
fi

cd "$TT_METAL_HOME"

# Set up host list for MPI commands
HOSTS="wh-glx-a04u02,wh-glx-a05u02"

# Step 1: Reset hardware using MPI
echo "=========================================="
echo "Step 1: Running tt-smi -glx_reset on all hosts..."
echo "=========================================="
mpirun --host $HOSTS --mca btl_tcp_if_exclude docker0,lo,tailscale0 tt-smi -glx_reset
if [[ $? -ne 0 ]]; then
    echo "Error: Hardware reset failed"
    exit 1
fi
sleep 5
echo ""

# Step 2: Run physical discovery test
echo "=========================================="
echo "Step 2: Running physical discovery test..."
echo "=========================================="
tt-run --rank-binding "$RANK_BINDING" --mpi-args "$MPI_ARGS_REVERSED" ./build/test/tt_metal/tt_fabric/test_physical_discovery --gtest_filter="PhysicalDiscovery.*"
if [[ $? -ne 0 ]]; then
    echo "Error: Physical discovery test failed"
    exit 1
fi
echo ""

# Test 1: test_all_to_all_dispatch_8x8_dual_galaxy
# To run all parametrized variants:
echo "Running test_all_to_all_dispatch_8x8_dual_galaxy (all variants)..."
tt-run --rank-binding "$RANK_BINDING" --mpi-args "$MPI_ARGS_REVERSED" bash -c "source ./python_env/bin/activate && pytest -svv tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x8_dual_galaxy"

# Or to run specific parametrized variants (matching the workflow):
# echo "Running test_all_to_all_dispatch_8x8_dual_galaxy (fabric_2d variant)..."
# tt-run --rank-binding "$RANK_BINDING" --mpi-args "$MPI_ARGS_REVERSED" bash -c "source ./python_env/bin/activate && pytest -svv \"tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x8_dual_galaxy[wormhole_b0-dram-dram-DataType.BFLOAT16-None-1-s2-7168-8-256-32-1-8x8_grid-False-fabric_2d]\""

# echo "Running test_all_to_all_dispatch_8x8_dual_galaxy (fabric_1d_line variant)..."
# tt-run --rank-binding "$RANK_BINDING" --mpi-args "$MPI_ARGS_REVERSED" bash -c "source ./python_env/bin/activate && pytest -svv \"tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x8_dual_galaxy[wormhole_b0-dram-dram-DataType.BFLOAT16-None-1-s2-7168-8-256-32-1-8x8_grid-False-fabric_1d_line]\""

# Test 2: test_all_to_all_combine_8x8_dual_galaxy
# To run all parametrized variants:
echo "Running test_all_to_all_combine_8x8_dual_galaxy (all variants)..."
tt-run --rank-binding "$RANK_BINDING" --mpi-args "$MPI_ARGS_REVERSED" bash -c "source ./python_env/bin/activate && pytest -svv tests/nightly/tg/ccl/test_all_to_all_combine_6U.py::test_all_to_all_combine_8x8_dual_galaxy"

# Or to run specific parametrized variant (matching the workflow):
# echo "Running test_all_to_all_combine_8x8_dual_galaxy (fabric_1d_line variant)..."
# tt-run --rank-binding "$RANK_BINDING" --mpi-args "$MPI_ARGS_REVERSED" bash -c "source ./python_env/bin/activate && pytest -svv \"tests/nightly/tg/ccl/test_all_to_all_combine_6U.py::test_all_to_all_combine_8x8_dual_galaxy[wormhole_b0-dram-dram-DataType.BFLOAT16-None-num_links_1-2-sparse-s2-7168-8-256-32-axis_1-8x8_grid-False-fabric_1d_line]\""
