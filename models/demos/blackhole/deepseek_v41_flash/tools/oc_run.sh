#!/bin/bash
# usage: oc_run.sh <logname> "<env assignments + command>"   (run ON the host; takes the device lock, resets devices first when OC_RESET=1, watches for hangs)
W=/mnt/tt-data/ssinghal/wt/pf_onecopy; L=/mnt/tt-data/ssinghal/dsv4-logs/pf_onecopy_$1.log; rm -f $L; touch $L
exec flock -w 14400 /tmp/dsv4_dev.lock bash -c "cd $W && source $W/python_env/bin/activate && export TT_METAL_CACHE=/mnt/tt-data/ssinghal/tt-metal-cache/oc\$(hostname -s | tr -dc 0-9 | tail -c 2) TT_METAL_HOME=$W PYTHONPATH=$W MOE_COMPUTE_FP32_ACC=1 MOE_COMPUTE_BFP8_WEIGHTS=1
 [ \"$OC_RESET\" = 1 ] && tt-smi -glx_reset > $L.reset 2>&1
 ( $2 ) > $L 2>&1 &
 pid=\$!; $W/models/demos/blackhole/deepseek_v41_flash/tools/hangwatch.sh \$pid $L ${OC_STALL:-8}; rc=\$?; wait \$pid; echo done rc=\$rc >> $L"
