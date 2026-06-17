#!/bin/bash
# BH master: run all four family sweeps -> conv_bench_data_bh_v2.csv (separate from WH's conv_bench_data.csv)
cd /localdev/wransom/tt-metal
export CB_CSV=/localdev/wransom/tt-metal/conv_bench_data_bh_v2.csv
echo "BH collection start: $(date)"
bash conv_bench_tools/cb_collect_rn50.sh
bash conv_bench_tools/cb_collect_sdxl.sh
bash conv_bench_tools/cb_collect_vu.sh
bash conv_bench_tools/cb_collect_vae.sh
echo "=== ALL BH COLLECTION COMPLETE === $(date)"
