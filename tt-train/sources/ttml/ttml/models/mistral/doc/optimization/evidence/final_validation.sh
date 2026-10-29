#!/bin/bash
# Final-path validation: loss trajectory vs the torch references (8 of 40 layers, real weights, host fp32 loss),
# allocator probe, and the full 40-layer TP=4 run with real weights.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
E=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/composite/evidence
PY=../python_env/bin/python
TRAIN=sources/examples/mistral_train/train.py
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== 8 layers real weights, host loss, final path $(date -u +%FT%TZ)"
timeout 3600 $PY $TRAIN --layers 8 --steps 10 --warmup-steps 2 --host-loss --loss-log $O/final_loss_8layers_hostloss.json > $O/final_train_8layers_hostloss.log 2>&1; echo "rc=$?"; grep -E "^step|layers," $O/final_train_8layers_hostloss.log | tail -3
$PY sources/examples/mistral_train/compare_loss.py --ttml $O/final_loss_8layers_hostloss.json --torch $E/loss_torch_8layers.json --torch-bf16 $E/loss_torch_8layers_bf16params.json --tol 0.10 > $O/final_loss_comparison.txt 2>&1; echo "compare rc=$?"; tail -4 $O/final_loss_comparison.txt
echo "=== memory probe 8 layers final path $(date -u +%FT%TZ)"
timeout 1800 $PY $TRAIN --layers 8 --steps 2 --warmup-steps 1 --random-init --random-data --mem-probe > $O/final_mem_probe_8layers.log 2>&1; grep "DRAM allocated" $O/final_mem_probe_8layers.log | tail -3
echo "=== TP=4 40 layers real weights, final path $(date -u +%FT%TZ)"
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
rm -f /dev/shm/sm_segment.* 2>/dev/null
sleep 300
timeout 7200 $PY $TRAIN --tp 4 --layers 0 --steps 10 --warmup-steps 2 --loss-log $O/final_full_tp4_real.json > $O/final_full_tp4_real.log 2>&1; echo "rc=$?"; grep -E "^step|layers," $O/final_full_tp4_real.log | tail -12
echo "=== done $(date -u +%FT%TZ)"
