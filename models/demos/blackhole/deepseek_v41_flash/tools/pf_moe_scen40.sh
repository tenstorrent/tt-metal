#!/bin/bash
# usage: scen40.sh <tag> <scen> [extra env...]  : 40-layer scenario test (prefill logits vs the CPU dump), unified prefill MoE, no decode weights
TAG=$1; SC=$2; shift; shift
cd /mnt/tt-data/ssinghal/wt/pf_moe
D=models/demos/blackhole/deepseek_v41_flash
exec $D/tools/devrun_pf.sh "env DSV41_LAYERS=0-39 DSV41_PREFILL_MOE=unified DSV41_UNI_NODECODE=1 DSV41_U=4 DSV41_REPS=2 DSV41_DIR_4096=/mnt/tt-data/ssinghal/dsv4-prefill-s4096b1f DSV41_DIR_8192=/mnt/tt-data/ssinghal/dsv4-prefill-s8192b1r 'DSV41_SCEN=$SC' DSV41_SAVE_LOGITS=/mnt/tt-data/ssinghal/dsv4-logs/pf_pf_moe_scen_$TAG $* timeout 43200 pytest -x -s -q $D/tests/test_prefill_scen_device.py"
