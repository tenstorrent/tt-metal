#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

export TT_LOGGER_LEVEL=FATAL
export TT_METAL_RUNTIME_ROOT=/home/tt-metal
export TT_METAL_HOME=/home/tt-metal

SCRIPT="/home/tt-metal/tt-train/build/sources/examples/nano_gpt/nano_gpt"
RESET_BOARD="tt-smi -r 0"
CONFIG="/home/tt-metal/tt-train/configs/training_shakespeare_nanogpt.yaml"
SLEEP_DURATION=30

$RESET_BOARD
echo "Running $SCRIPT..."

for i in {1..5}; do
    # Run with config file and unique run name
    $SCRIPT -c $CONFIG -n "run_$i" -t 1
    
    $RESET_BOARD
    echo "Sleeping for $SLEEP_DURATION seconds and restarting training..."
    sleep $SLEEP_DURATION
done

echo "Done running $SCRIPT"
