#!/bin/bash
# usage: mo_run40.sh <tag> <session ids with @a/@b/@c modes> [extra env...]   (run ON the host): 40-layer prefill-only session, unified MoE one-copy ring weights, in-process A/B modes
# modes: d = c + ring topology for dispatch/combine, e = ring topology only; a = baseline (unflagged), b = batched own-row router, c = batched router + shared expert || dispatch on sub-devices
TAG=$1; SS=$2; shift; shift
W=/mnt/tt-data/ssinghal/wt/pf_moeoverlap; D=models/demos/blackhole/deepseek_v41_flash
exec $W/$D/tools/mo_run.sh run40_$TAG "env DSV41_LAYERS=0-39 DSV41_PREFILL_MOE=unified DSV41_UNI_RING=1 DSV41_PREFILL_ONLY=1 DSV41_MEMLOG=1 DSV41_ENGRAM_RAM=1 DSV41_MO_OVERLAP=prep 'DSV41_MODE_A=DSV41_MO_OVERLAP=prep,DSV41_UNI_ROUTER=slices' 'DSV41_MODE_B=DSV41_MO_OVERLAP=prep,DSV41_UNI_ROUTER=batched' 'DSV41_MODE_C=DSV41_MO_OVERLAP=1,DSV41_UNI_ROUTER=batched' 'DSV41_MODE_D=DSV41_MO_OVERLAP=1,DSV41_UNI_ROUTER=batched,DSV41_UNI_TOPO=ring' 'DSV41_MODE_E=DSV41_MO_OVERLAP=prep,DSV41_UNI_ROUTER=slices,DSV41_UNI_TOPO=ring' 'DSV41_SESSION=$SS' $* timeout 20000 pytest -x -s -q -o junit_suite_name=mo_$TAG $D/demo/text_demo.py -k session"
