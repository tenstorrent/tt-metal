#!/bin/bash
# Setup environment for running GR00T N1.6 on Blackhole
# Source this file: source models/experimental/gr00t_n1.6/run_env.sh
# Then run Python scripts from the tt-metal directory

export TT_METAL_HOME=/home/ttuser/experiments/pi0/tt-metal
export ARCH_NAME=blackhole
export PYTHONPATH=/home/ttuser/experiments/gr00t_n16/tt-metal:$PYTHONPATH

echo "GR00T N1.6 environment set up"
echo "  TT_METAL_HOME=$TT_METAL_HOME"
echo "  ARCH_NAME=$ARCH_NAME"
echo "  Run Python from: cd /home/ttuser/experiments/gr00t_n16/tt-metal"
