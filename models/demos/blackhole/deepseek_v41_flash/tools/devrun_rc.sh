#!/bin/bash
# usage: devrun_rc.sh "<command>"  -- runs the command from the pf_reconf worktree on THIS host under the per-host device lock (queues behind other runs)
# Every DSV41_* / MOE_COMPUTE_* variable inherited from the shell is dropped first (no leftovers); pass the wanted ones inside the command (env VAR=...).
W=/mnt/tt-data/ssinghal/wt/pf_reconf
exec flock -w 14400 /tmp/dsv4_dev.lock bash -c "for v in \$(compgen -e | grep -E '^(DSV41_|MOE_COMPUTE_|TT_METAL_(CACHE|HOME))'); do unset \"\$v\"; done; cd $W && source $W/python_env/bin/activate && export TT_METAL_CACHE=/mnt/tt-data/ssinghal/tt-metal-cache/h$(hostname -s | tr -dc 0-9 | tail -c 2) TT_METAL_HOME=$W PYTHONPATH=$W MOE_COMPUTE_FP32_ACC=1 MOE_COMPUTE_BFP8_WEIGHTS=1 && $1"
