#!/bin/bash
# Command to run physical discovery test with 4 galaxies split into 16 partitions

# Set up variables
RANK_BINDING="tests/tt_metal/distributed/config/4galaxy_16partition_rank_bindings.yaml"
RANKFILE="$(pwd)/rankfile_4galaxy_16partition"
HOSTS="wh-glx-a04u02,wh-glx-a05u02,wh-glx-a05u08,wh-glx-a05u14"

# MPI arguments for multi-host communication
MPI_ARGS="--host $HOSTS --rankfile $RANKFILE --oversubscribe --mca btl self,tcp --mca btl_tcp_if_exclude docker0,lo,tailscale0 --bind-to none --tag-output"

# Ensure we're in TT_METAL_HOME
if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Error: TT_METAL_HOME is not set. Please set it to the root of your tt-metal repository."
    exit 1
fi

cd "$TT_METAL_HOME"

echo "=========================================="
echo "Running physical discovery test"
echo "Configuration: 4 galaxies, 16 partitions (4 partitions per galaxy)"
echo "=========================================="
echo ""

# Run physical discovery test
tt-run --rank-binding "$RANK_BINDING" --mpi-args "$MPI_ARGS" ./build/test/tt_metal/tt_fabric/test_physical_discovery --gtest_filter="PhysicalDiscovery.*"
