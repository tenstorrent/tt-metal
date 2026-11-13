#!/bin/bash
# go.sh TAG VAR=val ... : run run_attn.sh on .45 and print the interesting lines
D=/mnt/tt-data/ssinghal/wt/hy4_lattn/models/demos/blackhole/hy4_preview/tests
ssh 10.82.97.45 "cd $D && SCRIPT=${SCRIPT:-run_attn_device.py} ./run_attn.sh $* GOLD=/gold/golden" 2>&1 | grep -v Warning
L=/mnt/tt-data/ssinghal/hy4-logs/program/layer-attn/run_$1.log
grep -n "RES \|SUMMARY\|build \|Traceback\|Error\|TT_FATAL\|RC=" $L | cut -c1-600 | head -${N:-40}
