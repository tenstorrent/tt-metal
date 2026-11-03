#!/bin/bash
# Iteration 48 chain + the review round 14 P1 controls: harness, SwiGLU op tests, the single-chip 8-layer loss match,
# the TP=4 8-layer loss match vs torch with the overlap window / without / the serial control (the host loss from
# the gathered vocab shards), two 40-layer real-weights runs with the overlap (determinism) and one serial control,
# a long in-process alternating run (32 pairs).
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
E=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/composite/evidence
TAG=iter48
PY=../python_env/bin/python
TRAIN=sources/examples/mistral_train/train.py
MGD=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== ${TAG} harness $(date -u +%FT%TZ)"
timeout 3600 $PY -m pytest tests/python/test_mistral.py -q -p no:cacheprovider > $O/${TAG}_harness.log 2>&1; echo "harness rc=$?"; grep -E "passed|failed" $O/${TAG}_harness.log | tail -1
timeout 900 $PY -m pytest tests/python/test_swiglu_packed.py -q -p no:cacheprovider > $O/${TAG}_swiglu_op_tests.log 2>&1; echo "swiglu op tests rc=$?"; grep -E "passed|failed" $O/${TAG}_swiglu_op_tests.log | tail -1
echo "=== ${TAG} 8 layers real weights, host loss, one chip $(date -u +%FT%TZ)"
timeout 3600 $PY $TRAIN --layers 8 --steps 10 --warmup-steps 2 --host-loss --loss-log $O/${TAG}_loss_8layers_hostloss.json > $O/${TAG}_train_8layers_hostloss.log 2>&1; echo "rc=$?"
$PY sources/examples/mistral_train/compare_loss.py --ttml $O/${TAG}_loss_8layers_hostloss.json --torch $E/loss_torch_8layers.json --torch-bf16 $E/loss_torch_8layers_bf16params.json --tol 0.10 > $O/${TAG}_loss_comparison.txt 2>&1; echo "compare rc=$?"; grep -E "max \|diff\||PASS|FAIL" $O/${TAG}_loss_comparison.txt | head -2
export TT_MESH_GRAPH_DESC_PATH=$MGD
for v in overlap no_overlap serial; do
  rm -f /dev/shm/sm_segment.* 2>/dev/null
  extra=""; [ $v = no_overlap ] && extra="--no-ccl-overlap"; [ $v = serial ] && extra="--ccl-overlap-serial"
  echo "=== ${TAG} 8 layers real weights, host loss, TP=4 $v $(date -u +%FT%TZ)"
  timeout 3600 $PY $TRAIN --tp 4 --layers 8 --steps 10 --warmup-steps 2 --host-loss $extra --loss-log $O/${TAG}_loss_8layers_tp4_$v.json > $O/${TAG}_train_8layers_tp4_$v.log 2>&1; echo "rc=$?"
  $PY sources/examples/mistral_train/compare_loss.py --ttml $O/${TAG}_loss_8layers_tp4_$v.json --torch $E/loss_torch_8layers.json --torch-bf16 $E/loss_torch_8layers_bf16params.json --tol 0.10 > $O/${TAG}_loss_comparison_tp4_$v.txt 2>&1; echo "compare rc=$?"; grep -E "max \|diff\||PASS|FAIL" $O/${TAG}_loss_comparison_tp4_$v.txt | head -2
done
for v in overlap_a overlap_b serial; do
  rm -f /dev/shm/sm_segment.* 2>/dev/null
  extra=""; [ $v = serial ] && extra="--ccl-overlap-serial"
  echo "=== ${TAG} TP=4 40 layers real weights $v $(date -u +%FT%TZ)"; sleep 240
  timeout 7200 $PY $TRAIN --tp 4 --layers 0 --steps 10 --warmup-steps 2 --mem-probe $extra --loss-log $O/${TAG}_full_tp4_real_$v.json > $O/${TAG}_full_tp4_real_$v.log 2>&1; echo "rc=$?"; grep -E "layers,|overlap" $O/${TAG}_full_tp4_real_$v.log | tail -2
done
echo "=== ${TAG} alternating overlap A/B, 32 pairs $(date -u +%FT%TZ)"; sleep 300
rm -f /dev/shm/sm_segment.* 2>/dev/null
timeout 3600 $PY $TRAIN --tp 4 --layers 0 --steps 66 --warmup-steps 2 --random-init --random-data --ccl-overlap-alternate --loss-log $O/${TAG}_mfu_alternate.json > $O/${TAG}_mfu_alternate.log 2>&1; echo "rc=$?"; grep -E "layers," $O/${TAG}_mfu_alternate.log | tail -1
echo "=== ${TAG} chain finished $(date -u +%FT%TZ)"
