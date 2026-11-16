#!/bin/bash
# nl_b10.sh <tag> <NL> [env...] (RUN ON .44): whole-mesh run_single.py plain trace-only with perfstack flags on wt/hy4_b10
cd /mnt/tt-data/ssinghal/hy4-logs/program/b10; . ../perfstack/flags.sh; TAG=$1; NL=$2; shift 2
COMMON="NL=$NL MTP=0 HY4_OPT=all HY4_IHC_FUSED=1 HY4_FUSE=sparse,xchg HY4_XM_NSPLIT=8 HY4_SKINNY=shared,router,oproj,hproj TRACE_BYTES=700e6 HY4_PF_OPT=all,moeall TAGNAME=$TAG HY4_MOE_TP32=1 HY4_TP32_ONLY=1"
WD=${WD:-3000} ./run_b10.sh $TAG run_single.py $COMMON $PSFLAGS "$@"
