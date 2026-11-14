#!/bin/bash
# pfp_go.sh <tag> <script> [K=V...] : run on .45 (slot SLOT, default 2) and print the interesting lines of the log
D=/mnt/tt-data/ssinghal/wt/hy4_pfpool/models/demos/blackhole/hy4_preview/tests
tag=$1; script=$2; shift 2
ssh 10.82.97.45 "cd $D && SLOT=${SLOT:-2} CACHE=${CACHE:-} TMO=${TMO:-1800} ./pfp_run.sh $tag $script $*" 2>&1 | grep -v Warning
sleep 3
grep -a "RES \|Error\|Traceback\|TT_FATAL\|RC=" /mnt/tt-data/ssinghal/hy4-logs/program/pool-prefill/logs/run_$tag.log | cut -c1-${W:-700} | head -${N:-40}
