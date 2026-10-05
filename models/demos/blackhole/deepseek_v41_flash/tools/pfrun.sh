#!/bin/bash
# usage: pfrun.sh "<cmd>" -- per-host device lock; imports the package from the pf_host worktree
W=/mnt/tt-data/ssinghal/wt/pf_host; M=/mnt/tt-data/ssinghal/tests/tt-metal
exec flock -w 14400 /tmp/dsv4_dev.lock bash -c "cd $W && source $M/python_env/bin/activate && export TT_METAL_CACHE=/mnt/tt-data/ssinghal/tt-metal-cache/h\$(hostname -s | tr -dc 0-9 | tail -c 2) TT_METAL_HOME=$W PYTHONPATH=$W MOE_COMPUTE_FP32_ACC=1 MOE_COMPUTE_BFP8_WEIGHTS=1 && $1"
