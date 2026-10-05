#!/bin/bash
# usage: pfrun.sh <worktree> "<cmd>" -- runs the command from a private worktree (TT_METAL_HOME/PYTHONPATH = the worktree; build artifacts symlinked) under the per-host device lock
WT=${1:-/mnt/tt-data/ssinghal/wt/pf_attn}
exec flock -w 14400 /tmp/dsv4_dev.lock bash -c "cd $WT && source $WT/python_env/bin/activate && export TT_METAL_CACHE=/mnt/tt-data/ssinghal/tt-metal-cache/pf_\$(hostname -s | tr -dc 0-9 | tail -c 2) TT_METAL_HOME=$WT PYTHONPATH=$WT MOE_COMPUTE_FP32_ACC=1 MOE_COMPUTE_BFP8_WEIGHTS=1 && $2"
