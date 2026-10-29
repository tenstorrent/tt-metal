#!/bin/bash
# Iteration 4 (tail-write op + swept gate_up program config). Per the user's priority (2026-10-29): the full-model
# TP=4 MFU is the result; single-chip Tracy at 1/2/4/8 layers only attributes and ranks the next items. No
# single-chip MFU run, no 16-layer profile. Steps already completed in the aborted attempts (harness 14 passed
# -> iter4_harness.log, TP smoke -> iter4_tp_smoke.log, Tracy 1/2/4 -> iter4_tracy_{1,2,4}layers) are skipped
# when their artefacts exist.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
TRAIN=sources/examples/mistral_train/train.py
rm -f /dev/shm/sm_segment.* 2>/dev/null
if ! grep -q "passed" $O/iter4_harness.log 2>/dev/null; then
  echo "=== iter4 harness $(date -u +%FT%TZ)"
  timeout 3600 $PY -m pytest tests/python/test_mistral.py -q -p no:cacheprovider > $O/iter4_harness.log 2>&1; echo "harness rc=$?"
fi
grep -E "passed|failed" $O/iter4_harness.log | tail -1
if ! grep -q "layers," $O/iter4_tp_smoke.log 2>/dev/null; then
  echo "=== iter4 TP smoke 2 layers $(date -u +%FT%TZ)"
  export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
  timeout 1800 $PY $TRAIN --tp 4 --layers 2 --steps 3 --warmup-steps 1 --random-init --random-data > $O/iter4_tp_smoke.log 2>&1; echo "rc=$?"
  unset TT_MESH_GRAPH_DESC_PATH
fi
grep -E "transposed:|fused into|layers,|Error" $O/iter4_tp_smoke.log | tail -3
echo "=== iter4 tracy 1/2/4/8 (attribution) $(date -u +%FT%TZ)"
for L in 1 2 4 8; do
  if ls $O/iter4_tracy_${L}layers/reports/*/ops_perf_results*.csv > /dev/null 2>&1 && grep -q "OPs csv generated" $O/iter4_tracy_${L}layers.log; then
    echo "tracy $L already done"; continue
  fi
  rm -rf $O/iter4_tracy_${L}layers
  timeout 3600 $PY -m tracy -r -p --op-support-count 40000 -o $O/iter4_tracy_${L}layers $TRAIN --layers $L --steps 4 --warmup-steps 2 --random-init --random-data > $O/iter4_tracy_${L}layers.log 2>&1
  echo "tracy $L rc=$?"
done
COOL=600 $O/tp_measure.sh iter4_tp
echo "=== done $(date -u +%FT%TZ)"
