#!/bin/bash
# Iteration 47: the 8-layer real-weights loss trajectory at TP=4 against torch, with the collective overlap and without.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
E=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/composite/evidence
PY=../python_env/bin/python
TRAIN=sources/examples/mistral_train/train.py
for v in overlap no_overlap; do
  rm -f /dev/shm/sm_segment.* 2>/dev/null
  extra=""; [ $v = no_overlap ] && extra="--no-ccl-overlap"
  echo "=== $v $(date -u +%T)"
  timeout 3600 $PY $TRAIN --tp 4 --layers 8 --steps 10 --warmup-steps 2 --host-loss $extra --loss-log $O/iter47_loss_8layers_tp4_$v.json > $O/iter47_train_8layers_tp4_$v.log 2>&1; echo "rc=$?"
  $PY sources/examples/mistral_train/compare_loss.py --ttml $O/iter47_loss_8layers_tp4_$v.json --torch $E/loss_torch_8layers.json --torch-bf16 $E/loss_torch_8layers_bf16params.json --tol 0.10 > $O/iter47_loss_comparison_tp4_$v.txt 2>&1; echo "compare rc=$?"; grep -E "max \|diff\||PASS|FAIL" $O/iter47_loss_comparison_tp4_$v.txt | head -2
done
echo "=== tp loss check finished"
