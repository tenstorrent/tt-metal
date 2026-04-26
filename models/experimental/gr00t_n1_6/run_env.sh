#!/bin/bash
# Setup environment for running GR00T N1.6 on Blackhole
# Source this file: source models/experimental/groot_n16/run_env.sh
# Then run Python scripts from the pi0 tt-metal directory

export TT_METAL_HOME=/home/ttuser/experiments/gr00t_n16/tt-metal
export ARCH_NAME=blackhole
export PYTHONPATH=/home/ttuser/experiments/gr00t_n16/tt-metal:$PYTHONPATH
export GR00T_DEVICE_ID=${GR00T_DEVICE_ID:-2}

echo "GR00T N1.6 environment set up"
echo "  TT_METAL_HOME=$TT_METAL_HOME"
echo "  ARCH_NAME=$ARCH_NAME"
echo "  Run Python from: cd $TT_METAL_HOME"
