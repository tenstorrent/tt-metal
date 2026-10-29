#!/bin/bash
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
rm -f /dev/shm/sm_segment.* 2>/dev/null
timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 1 --out $O/tp_grad_check_single.npz > $O/tp_grad_check_single.log 2>&1; echo "single rc=$?"
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
rm -f /dev/shm/sm_segment.* 2>/dev/null
timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 4 --no-fuse --out $O/tp_grad_check_tp4_nofuse.npz --compare $O/tp_grad_check_single.npz > $O/tp_grad_check_tp4_nofuse.log 2>&1; echo "tp4 nofuse rc=$?"; grep -E "^tp |^  |PASS|FAIL|raw read" $O/tp_grad_check_tp4_nofuse.log
rm -f /dev/shm/sm_segment.* 2>/dev/null
timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 4 --out $O/tp_grad_check_tp4.npz --compare $O/tp_grad_check_single.npz > $O/tp_grad_check_tp4.log 2>&1; echo "tp4 fused rc=$?"; grep -E "^tp |^  |PASS|FAIL|raw read" $O/tp_grad_check_tp4.log
