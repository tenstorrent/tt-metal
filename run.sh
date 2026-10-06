#!/bin/bash
# usage (ON the device host): run.sh "<cmd>" -- per-host flock, env for the spec_adapt worktree (clean DSV41_* env, set the ones you need inside <cmd>)
W=/mnt/tt-data/ssinghal/wt/spec_int; M=/mnt/tt-data/ssinghal/tests/tt-metal
H=$(hostname -s | tr -dc 0-9 | tail -c 2)
exec flock -w 14400 /tmp/dsv4_dev.lock bash -c "
for v in \$(compgen -e | grep -E '^(DSV41_|MOE_COMPUTE_|TT_METAL_)'); do unset \$v; done
cd $W && source $M/python_env/bin/activate
export TT_METAL_CACHE=/mnt/tt-data/ssinghal/tt-metal-cache/h$H TT_METAL_HOME=$W PYTHONPATH=$W MOE_COMPUTE_FP32_ACC=1 MOE_COMPUTE_BFP8_WEIGHTS=1
$1"
