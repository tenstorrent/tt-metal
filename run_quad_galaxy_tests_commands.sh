#!/bin/bash
# Commands to run quad galaxy tests using tt-run
# Based on .github/workflows/multi-host-physical.yaml and tests/scripts/multihost/run_quad_galaxy_tests.sh

# Set up variables for quad galaxy
QUAD_RANK_BINDING="tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml"
QUAD_RANKFILE="$(pwd)/rankfile_quad_galaxy"
QUAD_HOSTS="wh-glx-a04u02,wh-glx-a05u02,wh-glx-a05u08,wh-glx-a05u14"
# For SLURM sessions with mpirun, using --rankfile with --host
# --host is needed for TCP BTL to establish connections between hosts
# --bind-to none is important for multi-host communication
# Using exclude instead of include - cnx1 interface doesn't exist on these hosts
QUAD_MPI_ARGS="--host $QUAD_HOSTS --rankfile $QUAD_RANKFILE --mca btl self,tcp --mca btl_tcp_if_exclude docker0,lo,tailscale0 --bind-to none --tag-output"

# Ensure we're in TT_METAL_HOME
if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Error: TT_METAL_HOME is not set. Please set it to the root of your tt-metal repository."
    exit 1
fi

cd "$TT_METAL_HOME"

# Step 1: Reset hardware for quad galaxy (all 4 hosts)
echo "=========================================="
echo "Step 1: Running tt-smi -glx_reset on all 4 hosts..."
echo "=========================================="
mpirun --host $QUAD_HOSTS --mca btl_tcp_if_exclude docker0,lo,tailscale0 tt-smi -glx_reset
if [[ $? -ne 0 ]]; then
    echo "Error: Hardware reset failed for quad galaxy"
    exit 1
fi
sleep 5
echo ""

# Step 2: Run physical discovery test (optional - uncomment if needed)
echo "=========================================="
echo "Step 2: Running physical discovery test..."
echo "=========================================="
tt-run --rank-binding "$QUAD_RANK_BINDING" --mpi-args "$QUAD_MPI_ARGS" ./build/test/tt_metal/tt_fabric/test_physical_discovery --gtest_filter="PhysicalDiscovery.*"
if [[ $? -ne 0 ]]; then
    echo "Error: Physical discovery test failed"
    exit 1
fi
echo ""

# Test: test_all_to_all_dispatch_8x16_quad_galaxy
echo "=========================================="
echo "Running test_all_to_all_dispatch_8x16_quad_galaxy (all variants)..."
echo "=========================================="
tt-run --rank-binding "$QUAD_RANK_BINDING" --mpi-args "$QUAD_MPI_ARGS" bash -c "source ./python_env/bin/activate && pytest -svv tests/nightly/tg/ccl/test_all_to_all_dispatch_6U.py::test_all_to_all_dispatch_8x16_quad_galaxy"
