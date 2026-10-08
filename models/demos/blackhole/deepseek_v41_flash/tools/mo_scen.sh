#!/bin/bash
# usage: mo_scen.sh <tag> <layers> <scen> [extra env assignments...]  (run ON the host): traced-chunk scenario test, unified prefill MoE, one-copy ring weights
TAG=$1; LY=$2; SC=$3; shift; shift; shift
W=/mnt/tt-data/ssinghal/wt/pf_moeoverlap; D=models/demos/blackhole/deepseek_v41_flash
exec $W/$D/tools/mo_run.sh scen_$TAG "env DSV41_LAYERS=$LY DSV41_PREFILL_MOE=unified DSV41_UNI_RING=1 DSV41_U=4 DSV41_REPS=2 DSV41_ENGRAM_RAM=0 DSV41_DIR_4096=/mnt/tt-data/ssinghal/dsv4-prefill-s4096b1f DSV41_DIR_8192=/mnt/tt-data/ssinghal/dsv4-prefill-s8192b1r 'DSV41_SCEN=$SC' DSV41_SAVE_LOGITS=/mnt/tt-data/ssinghal/dsv4-logs/pf_moeoverlap_scen_$TAG $* timeout 14000 pytest -x -s -q $D/tests/test_prefill_scen_device.py"
