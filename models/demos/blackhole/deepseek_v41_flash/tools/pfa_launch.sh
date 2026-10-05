#!/bin/bash
# usage: launch.sh <host-suffix> <logname> "<env + command>" [nohw]  (runs under the device lock on that host from the pf_attn worktree; hangwatch unless 4th arg = nohw, e.g. for queued launches)
H=$1; LOG=/mnt/tt-data/ssinghal/dsv4-logs/pf_pf_attn_$2.log; : > $LOG
T=/mnt/tt-data/ssinghal/wt/pf_attn/models/demos/blackhole/deepseek_v41_flash/tools
HW="setsid nohup bash -c 'sleep 60; until PID=\$(pgrep -n -f \"[p]ython_env/bin/python3 .*pytest\"); [ -n \"\$PID\" ]; do sleep 10; done; exec $T/hangwatch.sh \$PID $LOG ${STALL:-20}' > $LOG.hw 2>&1 < /dev/null &"
[ -n "$4" ] && HW=""
ssh -o BatchMode=yes 10.82.97.$H "setsid nohup $T/pfrun.sh /mnt/tt-data/ssinghal/wt/pf_attn '$3' > $LOG 2>&1 < /dev/null &
$HW" 2>&1 | grep -v Warning
echo launched $LOG
