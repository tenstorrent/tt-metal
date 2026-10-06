#!/bin/bash
# usage: run40.sh <scenario> <tag> [extra env...]   (run on the host that should execute it)
SC=$1; TAG=$2; shift; shift
cd /mnt/tt-data/ssinghal/wt/pf_moe
D=models/demos/blackhole/deepseek_v41_flash
exec $D/tools/devrun_pf.sh "env DSV41_LAYERS=0-39 DSV41_PREFILL_MOE=unified DSV41_UNI_NODECODE=1 DSV41_PREFILL_ONLY=1 DSV41_MEMLOG=1 DSV41_ENGRAM_RAM=1 DSV41_SESSION=$SC $* timeout 43200 pytest -x -s -q -o junit_suite_name=pf_moe_$TAG $D/demo/text_demo.py -k session"
