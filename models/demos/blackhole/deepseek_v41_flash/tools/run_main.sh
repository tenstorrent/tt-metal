#!/bin/bash
# usage: run_main.sh "<cmd>" -- like run.sh but runs from the MAIN tree (no overlay), under the device lock
exec flock -w 14400 /tmp/dsv4_dev.lock bash -c "cd /mnt/tt-data/ssinghal/tests/tt-metal && source python_env/bin/activate && export TT_METAL_CACHE=/mnt/tt-data/ssinghal/tt-metal-cache/h44 TT_METAL_HOME=/mnt/tt-data/ssinghal/tests/tt-metal PYTHONPATH=/mnt/tt-data/ssinghal/tests/tt-metal MOE_COMPUTE_FP32_ACC=1 MOE_COMPUTE_BFP8_WEIGHTS=1 && $1"
