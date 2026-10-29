#!/bin/bash
# Iterations 5b (batched TP grads also for the composite TP adapters: qkv, out) and 6 (all-core RMSNorm backward):
# harness, TP=4 gradient check (both on), single-chip 8-layer Tracy for the RMSNorm attribution (iteration 6 on),
# TP=4 measurement of 5b alone (--no-allcore-rmsnorm), then of 5b + 6, each after a 600 s cool-down.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
TRAIN=sources/examples/mistral_train/train.py
MGD=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== iter6 harness $(date -u +%FT%TZ)"
timeout 3600 $PY -m pytest tests/python/test_mistral.py -q -p no:cacheprovider > $O/iter6_harness.log 2>&1; echo "harness rc=$?"; grep -E "passed|failed" $O/iter6_harness.log | tail -1
echo "=== iter5b+6 TP=4 gradient check $(date -u +%FT%TZ)"
export TT_MESH_GRAPH_DESC_PATH=$MGD; rm -f /dev/shm/sm_segment.* 2>/dev/null
timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 4 --out $O/iter6_tp_grad_check_tp4.npz --compare $O/tp_grad_check_single.npz > $O/iter6_tp_grad_check_tp4.log 2>&1; echo "tp4 rc=$?"; grep -E "^tp |PASS|FAIL|partial|RMSNorm" $O/iter6_tp_grad_check_tp4.log
unset TT_MESH_GRAPH_DESC_PATH
echo "=== iter6 single-chip Tracy 8 layers $(date -u +%FT%TZ)"
rm -rf $O/iter6_tracy_8layers
timeout 3600 $PY -m tracy -r -p --op-support-count 40000 -o $O/iter6_tracy_8layers $TRAIN --layers 8 --steps 4 --warmup-steps 2 --random-init --random-data > $O/iter6_tracy_8layers.log 2>&1; echo "tracy 8 rc=$?"
EXTRA_ARGS="--no-allcore-rmsnorm" COOL=600 $O/tp_measure.sh iter5b_tp
COOL=600 $O/tp_measure.sh iter6_tp
echo "=== done $(date -u +%FT%TZ)"
