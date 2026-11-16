#!/bin/bash
# prof_b10.sh <tag> [env...] (RUN ON .44): device profile of the trace replay (NLMAX=3: L0 dense, L1 indexer MoE, L2 MoE), current perf stack (perfstack + ATTN_FUSE + KV_FUSE + TP24 east)
cd /mnt/tt-data/ssinghal/hy4-logs/program/b10; . ../perfstack/flags.sh; TAG=$1; shift
. ../perfstack/flags.sh
COMMON="HY4_OPT=all HY4_IHC_FUSED=1 HY4_FUSE=sparse,xchg HY4_XM_NSPLIT=8 HY4_SKINNY=shared,router,oproj,hproj TRACE_BYTES=700e6 HY4_PF_OPT=all,moeall TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1"
WD=${WD:-2400} ./run_b10.sh $TAG recon_prof.py $COMMON NLMAX=3 SEL=1,2 MODE=topk CTX=65536 STOCK_STEP=1 HY4_FUSE_AR=1 HY4_MOE_TP32=1 HY4_TP32_ONLY=1 HY4_TP32_PACK=u16 HY4_TP32_ROUTE=local HY4_TP32_TAIL=oneshot TAGNAME=$TAG $PSFLAGS HY4_ATTN_FUSE=1 HY4_TP24_PLACE=east HY4_KV_FUSE=1 "$@"
