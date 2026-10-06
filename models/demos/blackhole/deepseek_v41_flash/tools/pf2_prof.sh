#!/bin/bash
# usage: pf2_prof.sh <tag> "<PROF_* / DSV41_* env ...>"   (run ON the target host; holds the per-host device lock)
# whole-layer profile of one traced prefill chunk (tests/test_prefill_prof.py); profiler dir dsv4-logs/pf_pf_prof2_<tag>; analysis: tools/prof2_sum.py
tag=$1; envs=$2
W=/mnt/tt-data/ssinghal/wt/pf_prof2
O=/mnt/tt-data/ssinghal/dsv4-logs/pf_pf_prof2_$tag; rm -rf $O; mkdir -p $O
exec $W/models/demos/blackhole/deepseek_v41_flash/tools/pfrun.sh "for v in \$(compgen -e | grep -E '^(DSV41_|PROF_)'); do unset \$v; done; export DSV41_PREFILL_MOE=unified DSV41_UNI_NODECODE=1 DSV41_PREFILL_OPT=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_DIR=$O $envs; timeout 5400 pytest -x -s -q models/demos/blackhole/deepseek_v41_flash/tests/test_prefill_prof.py > $O.log 2>&1" $W
