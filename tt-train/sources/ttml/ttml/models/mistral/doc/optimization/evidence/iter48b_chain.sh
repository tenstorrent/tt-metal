#!/bin/bash
# Iteration 48b: the chain on the fixed overlap window (persistent staging / output buffers, rotating semaphore sets):
# TP gradient checks (1 block vs single chip; 2 blocks on / off), two 40-layer real-weights runs with the window and
# the serial control (determinism, the window == the serial routes), the 32-pair in-process alternating A/B, the
# cooled TP=4 MFU + 4-layer profile, the harness and the SwiGLU op tests.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
TAG=iter48b
PY=../python_env/bin/python
TRAIN=sources/examples/mistral_train/train.py
MGD=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
export TT_MESH_GRAPH_DESC_PATH=$MGD
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== ${TAG} TP=4 gradient check $(date -u +%FT%TZ)"
timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 4 --out $O/${TAG}_tp_grad_check_tp4.npz --compare $O/tp_grad_check_single.npz > $O/${TAG}_tp_grad_check_tp4.log 2>&1; echo "tp4 rc=$?"; grep -E "^tp |PASS|FAIL|Error" $O/${TAG}_tp_grad_check_tp4.log | head -4
for v in overlap no_overlap; do
  rm -f /dev/shm/sm_segment.* 2>/dev/null; extra=""; [ $v = no_overlap ] && extra="--no-ccl-overlap"
  timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 4 --layers 2 $extra --out $O/${TAG}_tp_grad_check_2layers_$v.npz > $O/${TAG}_tp_grad_check_2layers_$v.log 2>&1; echo "2 layers $v rc=$?"
done
python3 - > $O/${TAG}_tp_grad_check_2layers_compare.txt <<'PYEOF'
import numpy as np
O = "/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence"
a, b = np.load(f"{O}/iter48b_tp_grad_check_2layers_overlap.npz"), np.load(f"{O}/iter48b_tp_grad_check_2layers_no_overlap.npz")
worst = 0.0
print("2-block TP=4 gradient check, the overlap window on vs off (iteration 48b): relative L2 difference per gradient")
for k in sorted(a.files):
    if k.startswith("param:"): continue
    x, y = a[k].astype(np.float64), b[k].astype(np.float64)
    rel = np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-30); worst = max(worst, rel)
    print(f"{k:55s} {str(x.shape):20s} rel {rel:.2e}")
print("PASS" if worst < 1e-3 else "FAIL", f"(worst {worst:.2e}; the gate_up / qkv A gradients take the owned product in the window: rounding-level)")
PYEOF
tail -1 $O/${TAG}_tp_grad_check_2layers_compare.txt
for v in overlap_a overlap_b serial; do
  rm -f /dev/shm/sm_segment.* 2>/dev/null; extra=""; [ $v = serial ] && extra="--ccl-overlap-serial"
  echo "=== ${TAG} TP=4 40 layers real weights $v $(date -u +%FT%TZ)"; sleep 180
  timeout 7200 $PY $TRAIN --tp 4 --layers 0 --steps 10 --warmup-steps 2 --mem-probe $extra --loss-log $O/${TAG}_full_tp4_real_$v.json > $O/${TAG}_full_tp4_real_$v.log 2>&1; echo "rc=$?"; grep -E "layers,|overlap" $O/${TAG}_full_tp4_real_$v.log | tail -2
done
python3 - > $O/${TAG}_real_trajectories.txt <<'PYEOF'
import json
O = "/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence"
runs = {n: {e["step"]: e["loss"] for e in json.load(open(f"{O}/{f}"))["steps"]} for n, f in [("46", "iter46_full_tp4_real.json"), ("48b overlap a", "iter48b_full_tp4_real_overlap_a.json"), ("48b overlap b", "iter48b_full_tp4_real_overlap_b.json"), ("48b serial", "iter48b_full_tp4_real_serial.json")]}
for s in range(1, 11):
    print(s, "  ".join(f"{n}: {runs[n][s]:.4f}" for n in runs))
a, b, c = (runs[k] for k in ("48b overlap a", "48b overlap b", "48b serial"))
print("determinism (a == b):", "PASS" if all(a[s] == b[s] for s in a) else "FAIL")
print("window == serial routes:", "PASS" if all(a[s] == c[s] for s in a) else "FAIL (max |diff| %.4f)" % max(abs(a[s] - c[s]) for s in a))
PYEOF
tail -2 $O/${TAG}_real_trajectories.txt
echo "=== ${TAG} alternating overlap A/B, 32 pairs $(date -u +%FT%TZ)"; sleep 240
rm -f /dev/shm/sm_segment.* 2>/dev/null
timeout 3600 $PY $TRAIN --tp 4 --layers 0 --steps 66 --warmup-steps 2 --random-init --random-data --ccl-overlap-alternate --loss-log $O/${TAG}_mfu_alternate.json > $O/${TAG}_mfu_alternate.log 2>&1; echo "rc=$?"; grep -E "layers," $O/${TAG}_mfu_alternate.log | tail -1
unset TT_MESH_GRAPH_DESC_PATH
COOL=600 $O/tp_measure.sh ${TAG}_tp
CSV=$(ls $O/${TAG}_tp_tracy_tp4_4layers/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_rank_ops.py "$CSV" --devices 0,2 --top 40 > $O/${TAG}_tp_ranked_ops.txt 2>&1
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_rank_ops.py "$CSV" --devices 0,2 --top 400 > $O/${TAG}_tp_ranked_ops_full.txt 2>&1
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_analysis.py "$CSV" > $O/${TAG}_tp_analysis.txt 2>&1
python3 $O/cycle_diff.py $O/iter47_tp_ranked_ops_full.txt $O/${TAG}_tp_ranked_ops_full.txt --label "47 -> 48b" --min 0.02 > $O/${TAG}_cycle_diff.txt
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== ${TAG} harness $(date -u +%FT%TZ)"
timeout 3600 $PY -m pytest tests/python/test_mistral.py -q -p no:cacheprovider > $O/${TAG}_harness.log 2>&1; echo "harness rc=$?"; grep -E "passed|failed" $O/${TAG}_harness.log | tail -1
timeout 900 $PY -m pytest tests/python/test_swiglu_packed.py -q -p no:cacheprovider > $O/${TAG}_swiglu_op_tests.log 2>&1; echo "swiglu op tests rc=$?"; grep -E "passed|failed" $O/${TAG}_swiglu_op_tests.log | tail -1
echo "=== ${TAG} chain finished $(date -u +%FT%TZ)"
