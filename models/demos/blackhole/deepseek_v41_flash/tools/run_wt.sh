#!/bin/bash
# usage: run_wt.sh "<cmd>" -- runs from the pf_mhc worktree under the per-host device lock
W=/mnt/tt-data/ssinghal/wt/pf_mhc
exec flock -w 14400 /tmp/dsv4_dev.lock bash -c "cd $W && source $W/python_env/bin/activate && export TT_METAL_CACHE=/mnt/tt-data/ssinghal/tt-metal-cache/pfmhc_h\$(hostname -s | tr -dc 0-9 | tail -c 2) TT_METAL_HOME=$W PYTHONPATH=$W MOE_COMPUTE_FP32_ACC=1 MOE_COMPUTE_BFP8_WEIGHTS=1 && $1"
