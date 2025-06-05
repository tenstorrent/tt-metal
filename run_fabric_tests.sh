#!/bin/bash

# Export TT Metal environment variables
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=/home/ttuser/git/tt-metal
export PYTHONPATH=/home/ttuser/git/tt-metal
export TT_METAL_ENV=dev
export TT_HOST_RANK=0
export TT_MESH_ID=1
#Display environment information
echo "=== Environment Variables on $(hostname) ==="
echo "ARCH_NAME: $ARCH_NAME"
echo "TT_METAL_HOME: $TT_METAL_HOME"
echo "PYTHONPATH: $PYTHONPATH"
echo "TT_METAL_ENV: $TT_METAL_ENV"
echo "TT_MESH_ID: $TT_MESH_ID"
echo "TT_HOST_RANK: $TT_HOST_RANK"
echo "Current working directory: $(pwd)"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "============================================="

# Run the fabric unit tests
$TT_METAL_HOME/build/test/tt_metal/multi_host_fabric_tests |& tee log.txt
