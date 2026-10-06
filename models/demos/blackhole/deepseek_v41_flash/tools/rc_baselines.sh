#!/bin/bash
# usage (run on the device host): rc_baselines.sh <layers> "<scenario ids>" <tag> [extra env assignments...]
# one FRESH process per scenario (the old single-build behaviour), logs pf_reconf_<tag>_<scenario>.log
L=$1; IDS=$2; TAG=$3; shift 3
D=models/demos/blackhole/deepseek_v41_flash
for s in $IDS; do
  /mnt/tt-data/ssinghal/wt/pf_reconf/$D/tools/devrun_rc.sh "env DSV41_LAYERS=$L DSV41_ENGRAM_RAM=1 DSV41_MEMLOG=1 DSV41_SESSION=$s $* timeout 5400 pytest -x -s -q -o junit_suite_name=pf_reconf_$TAG $D/demo/text_demo.py -k session" > /mnt/tt-data/ssinghal/dsv4-logs/pf_reconf_${TAG}_$s.log 2>&1
done
