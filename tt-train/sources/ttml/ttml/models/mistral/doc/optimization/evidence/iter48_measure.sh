#!/bin/bash
# Iteration 48 measurement: cooled TP=4 MFU + 4-layer profile with the attention shards on the fused path (two overlap
# windows per layer), ranked tables, the cycle diff against iteration 47.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
unset TT_MESH_GRAPH_DESC_PATH
COOL=480 $O/tp_measure.sh iter48_tp
CSV=$(ls $O/iter48_tp_tracy_tp4_4layers/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_rank_ops.py "$CSV" --devices 0,2 --top 40 > $O/iter48_tp_ranked_ops.txt 2>&1
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_rank_ops.py "$CSV" --devices 0,2 --top 400 > $O/iter48_tp_ranked_ops_full.txt 2>&1
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_analysis.py "$CSV" > $O/iter48_tp_analysis.txt 2>&1
python3 $O/cycle_diff.py $O/iter47_tp_ranked_ops_full.txt $O/iter48_tp_ranked_ops_full.txt --label "47 -> 48" --min 0.02 > $O/iter48_cycle_diff.txt
grep -E "overlap" $O/iter48_tp_mfu_tp4.log | head -3
echo "=== iter48 measurements finished $(date -u +%FT%TZ)"
