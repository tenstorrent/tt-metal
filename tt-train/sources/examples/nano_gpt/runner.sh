# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash

export TT_METAL_LOGGER_LEVEL=FATAL
SCRIPT="/home/ubuntu/ML-Framework-CPP/build/sources/examples/nano_gpt/nano_gpt"
RESET_BOARD="tt-smi -r 0"
INTERVAL=100
DEFAULT_SEED=5489
MAX_STEPS=5000
SLEEP_DURATION=30

$RESET_BOARD
echo "Running $SCRIPT..."
for i in {1..5}; do
    $SCRIPT -i $INTERVAL -p transformer.msgpack -s $((DEFAULT_SEED - i)) -m $MAX_STEPS
    $RESET_BOARD
    echo "Sleeping for $SLEEP_DURATION seconds and restarting training..."
    sleep $SLEEP_DURATION
done
echo "Done running $SCRIPT"
