#!/bin/bash

source python_env/bin/activate
cd /home/ubuntu/Workspace/main_repo/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH="${TT_METAL_HOME}"

export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
export MESH_DEVICE=N150

echo "Environment activated for Main repo"
