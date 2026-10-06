#!/bin/bash
# usage: WT=<worktree> integ_launch.sh <host-suffix> <tag> "<flags>" <sessions> [ab]  -- starts integ_e2e.sh on 10.82.97.<host> under the device lock + hangwatch; log dsv4-logs/pf_pf_integ_e2e_<tag>.log
H=$1; T=$2; WT=${WT:-$(cd "$(dirname "$0")/../../../../.." && pwd)}; LOG=/mnt/tt-data/ssinghal/dsv4-logs/pf_pf_integ_e2e_$T.log
ssh -o BatchMode=yes 10.82.97.$H "cd $WT && setsid nohup models/demos/blackhole/deepseek_v41_flash/tools/pfrun.sh 'models/demos/blackhole/deepseek_v41_flash/tools/integ_e2e.sh $T \"$3\" $4 \"$5\"' > $LOG 2>&1 < /dev/null &"
ssh -o BatchMode=yes 10.82.97.$H "setsid nohup bash -c 'until PID=\$(pgrep -n -f \"[p]ython_env/bin/python3 .*junit_suite_name=integ_$T\"); [ -n \"\$PID\" ]; do sleep 10; done; sleep 20; exec /mnt/tt-data/ssinghal/hangwatch.sh \$PID $LOG 45' > $LOG.hw 2>&1 < /dev/null &"
