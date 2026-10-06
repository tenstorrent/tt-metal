#!/bin/bash
# usage (run on the device host): rc_session.sh <layers> "<scenario ids comma separated>" <tag> [extra env assignments...]
# ONE process, a multi-batch session: the model is built once and reconfigured between batch sizes (DSV41_SESSION). Log: pf_reconf_<tag>.log
L=$1; IDS=$2; TAG=$3; shift 3
D=models/demos/blackhole/deepseek_v41_flash
/mnt/tt-data/ssinghal/wt/pf_reconf/$D/tools/devrun_rc.sh "env DSV41_LAYERS=$L DSV41_ENGRAM_RAM=1 DSV41_MEMLOG=1 DSV41_SESSION=$IDS $* timeout 14000 pytest -x -s -q -o junit_suite_name=pf_reconf_$TAG $D/demo/text_demo.py -k session" > /mnt/tt-data/ssinghal/dsv4-logs/pf_reconf_$TAG.log 2>&1
