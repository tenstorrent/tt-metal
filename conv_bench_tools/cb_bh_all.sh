#!/bin/bash
# BH master (UNIFIED non-pin branch): run all four family sweeps -> conv_bench_data_unify.csv
WT=/localdev/wransom/tt-metal/.claude/worktrees/agent-a48fa14207415d0cb
cd "$WT"
export CB_CSV="$WT/conv_bench_data_unify.csv"
echo "BH UNIFIED collection start: $(date)"
bash "$WT/conv_bench_tools/cb_collect_rn50.sh"
bash "$WT/conv_bench_tools/cb_collect_sdxl.sh"
bash "$WT/conv_bench_tools/cb_collect_vu.sh"
bash "$WT/conv_bench_tools/cb_collect_vae.sh"
echo "=== ALL BH UNIFIED COLLECTION COMPLETE === $(date)"
