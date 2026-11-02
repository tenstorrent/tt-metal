#!/bin/bash
# Iteration 47 measurement: cooled TP=4 MFU + 4-layer profile with the collective overlap (A), the same MFU run with
# --no-ccl-overlap (B), then A again (ABA, 10-minute cool-downs); ranked tables and the cycle diff against 46c.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
TRAIN=sources/examples/mistral_train/train.py
MGD=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
sample() {
  while [ ! -f "$2" ]; do
    echo "$(date -u +%T) $(timeout 5 tt-smi -s 2>/dev/null | $PY -c "
import sys, json
try:
    devs = json.load(sys.stdin)['device_info']
    print(' '.join(f\"chip{i}: {d['telemetry']['aiclk'].strip()}MHz {d['telemetry']['asic_temperature'].strip()}C {d['telemetry']['power'].strip()}W\" for i, d in enumerate(devs[:4])))
except Exception: print('n/a')")"
    sleep 1
  done > "$1"
}
mfu() {  # name, cool, extra args
  local name=$1 cool=$2; shift 2
  export TT_MESH_GRAPH_DESC_PATH=$MGD
  rm -f /dev/shm/sm_segment.* 2>/dev/null
  echo "=== $name after ${cool}s cool-down $(date -u +%FT%TZ)"; sleep $cool
  rm -f $O/iter47.flag; sample $O/${name}_tt-smi.txt $O/iter47.flag &
  timeout 3600 $PY $TRAIN "$@" --tp 4 --layers 0 --steps 8 --warmup-steps 2 --random-init --random-data --loss-log $O/${name}.json > $O/${name}.log 2>&1; echo "rc=$?"; touch $O/iter47.flag; wait
  grep -E "layers,|overlap" $O/${name}.log | tail -3
}
unset TT_MESH_GRAPH_DESC_PATH
COOL=600 $O/tp_measure.sh iter47_tp
CSV=$(ls $O/iter47_tp_tracy_tp4_4layers/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_rank_ops.py "$CSV" --devices 0,2 --top 40 > $O/iter47_tp_ranked_ops.txt 2>&1
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_rank_ops.py "$CSV" --devices 0,2 --top 400 > $O/iter47_tp_ranked_ops_full.txt 2>&1
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_analysis.py "$CSV" > $O/iter47_tp_analysis.txt 2>&1
python3 $O/cycle_diff.py $O/iter46c_tp_ranked_ops_full.txt $O/iter47_tp_ranked_ops_full.txt --label "46c -> 47" --min 0.02 > $O/iter47_cycle_diff.txt
grep -E "overlap" $O/iter47_tp_mfu_tp4.log | head -3
mfu iter47_mfu_no_overlap 600 --no-ccl-overlap
mfu iter47_mfu_overlap2 600
echo "=== iter47 measurements finished $(date -u +%FT%TZ)"
