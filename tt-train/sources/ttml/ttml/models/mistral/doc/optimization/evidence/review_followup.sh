#!/bin/bash
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== TP gradient check: single chip reference $(date -u +%FT%TZ)"
timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 1 --out $O/tp_grad_check_single.npz > $O/tp_grad_check_single.log 2>&1; echo "rc=$?"; grep -E "^tp |saved" $O/tp_grad_check_single.log
echo "=== 16-layer Tracy, final path (plan rule: 16 fits now) $(date -u +%FT%TZ)"
rm -rf $O/iter3_fused_tracy_16layers
timeout 3600 $PY -m tracy -r -p --op-support-count 60000 -o $O/iter3_fused_tracy_16layers sources/examples/mistral_train/train.py --layers 16 --steps 4 --warmup-steps 2 --random-init --random-data --mem-probe > $O/iter3_fused_tracy_16layers.log 2>&1; echo "rc=$?"; grep -E "layers,|DRAM allocated after forward" $O/iter3_fused_tracy_16layers.log | tail -2
echo "=== TP gradient check: TP=4 $(date -u +%FT%TZ)"
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
rm -f /dev/shm/sm_segment.* 2>/dev/null
timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 4 --out $O/tp_grad_check_tp4.npz --compare $O/tp_grad_check_single.npz > $O/tp_grad_check_tp4.log 2>&1; echo "rc=$?"; grep -E "^tp |saved|^  |PASS|FAIL" $O/tp_grad_check_tp4.log
echo "=== done $(date -u +%FT%TZ)"
