#!/bin/bash
# usage: devrun.sh "<command>"  -- runs the command with the dsv4 env on this host, holding the device lock (queues behind other runs)
exec flock -w 3600 /tmp/dsv4_dev.lock bash -c "cd /mnt/tt-data/ssinghal/tests/tt-metal && source python_env/bin/activate && export TT_METAL_CACHE=/mnt/tt-data/ssinghal/tt-metal-cache/h44 TT_METAL_HOME=\$PWD PYTHONPATH=\$PWD MOE_COMPUTE_FP32_ACC=1 MOE_COMPUTE_BFP8_WEIGHTS=1 && $1"
