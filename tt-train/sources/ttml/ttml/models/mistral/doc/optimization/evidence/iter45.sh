#!/bin/bash
# Iteration 45 (metal trace replay of the whole TP=4 step with AdamW's bias-correction scalars computed on the device:
# TTML_ADAMW_DEVICE_STEP=1, --trace / --trace-alternate, TTML_TRACE_REGION_SIZE 1 GiB).
# Measurements first on the cool box (eager device-scalars, trace, alternate, eager again: ABBA-ish), then the
# correctness chain (harness, SwiGLU op tests, TP gradient check, 8-layer loss match vs torch, real weights trace).
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
export TTML_ADAMW_DEVICE_STEP=1 TTML_TRACE_REGION_SIZE=1073741824
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
E=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/composite/evidence
TAG=${TAG:-iter45}
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
run40() {  # name, cool seconds, train args...
  local name=$1 cool=$2; shift 2
  rm -f /dev/shm/sm_segment.* 2>/dev/null
  echo "=== ${TAG} $name after ${cool}s cool-down $(date -u +%FT%TZ)"; sleep $cool
  rm -f $O/${TAG}.flag; sample $O/${TAG}_${name}_tt-smi.txt $O/${TAG}.flag &
  timeout 3600 $PY $TRAIN --tp 4 --layers 0 --random-init --random-data "$@" --loss-log $O/${TAG}_${name}.json > $O/${TAG}_${name}.log 2>&1; echo "rc=$?"; touch $O/${TAG}.flag; wait
  grep -E "layers,|trace|capture" $O/${TAG}_${name}.log | grep -vE "^step" | tail -2
}
export TT_MESH_GRAPH_DESC_PATH=$MGD
run40 eager_tp4 0 --steps 8 --warmup-steps 2
run40 trace_tp4 600 --steps 10 --warmup-steps 2 --trace
run40 alternate_tp4 600 --steps 12 --warmup-steps 2 --trace --trace-alternate
run40 eager2_tp4 600 --steps 8 --warmup-steps 2
run40 trace2_tp4 600 --steps 10 --warmup-steps 2 --trace
echo "=== ${TAG} measurements finished $(date -u +%FT%TZ)"
$PY - <<'PYEOF' > $O/${TAG}_loss_eager_vs_trace.txt 2>&1
import json
O = "/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence"
runs = {n: json.load(open(f"{O}/iter45_{n}.json")) for n in ("eager_tp4", "trace_tp4", "alternate_tp4", "eager2_tp4", "trace2_tp4")}
def rows(r): return r["steps"] if isinstance(r, dict) and "steps" in r else r
ref = {e["step"]: e["loss"] for e in rows(runs["eager_tp4"])}
ok = True
for n, r in runs.items():
    for e in rows(r):
        if e["step"] in ref:
            d = abs(e["loss"] - ref[e["step"]])
            flag = "" if d < 1e-6 else "  <-- differs"
            ok &= d < 1e-6
            print(f"{n:14s} step {e['step']:2d} loss {e['loss']:.4f} ms {e['ms']:7.1f} trace={e.get('trace')} {flag}")
print("PASS" if ok else "FAIL", "(every run's losses identical to the eager run step for step)")
PYEOF
tail -1 $O/${TAG}_loss_eager_vs_trace.txt
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== ${TAG} harness (device step scalars) $(date -u +%FT%TZ)"
timeout 3600 $PY -m pytest tests/python/test_mistral.py -q -p no:cacheprovider > $O/${TAG}_harness.log 2>&1; echo "harness rc=$?"; grep -E "passed|failed" $O/${TAG}_harness.log | tail -1
timeout 900 $PY -m pytest tests/python/test_swiglu_packed.py -q -p no:cacheprovider > $O/${TAG}_swiglu_op_tests.log 2>&1; echo "swiglu op tests rc=$?"; grep -E "passed|failed" $O/${TAG}_swiglu_op_tests.log | tail -1
echo "=== ${TAG} 8 layers real weights, host loss, device step scalars $(date -u +%FT%TZ)"
timeout 3600 $PY $TRAIN --layers 8 --steps 10 --warmup-steps 2 --host-loss --loss-log $O/${TAG}_loss_8layers_hostloss.json > $O/${TAG}_train_8layers_hostloss.log 2>&1; echo "rc=$?"
$PY sources/examples/mistral_train/compare_loss.py --ttml $O/${TAG}_loss_8layers_hostloss.json --torch $E/loss_torch_8layers.json --torch-bf16 $E/loss_torch_8layers_bf16params.json --tol 0.10 > $O/${TAG}_loss_comparison.txt 2>&1; echo "compare rc=$?"; tail -3 $O/${TAG}_loss_comparison.txt
echo "=== ${TAG} TP=4 gradient check $(date -u +%FT%TZ)"
export TT_MESH_GRAPH_DESC_PATH=$MGD
rm -f /dev/shm/sm_segment.* 2>/dev/null
timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 4 --out $O/${TAG}_tp_grad_check_tp4.npz --compare $O/tp_grad_check_single.npz > $O/${TAG}_tp_grad_check_tp4.log 2>&1; echo "tp4 rc=$?"; grep -E "^tp |PASS|FAIL|Error" $O/${TAG}_tp_grad_check_tp4.log | head -6
echo "=== ${TAG} TP=4 40 layers real weights, trace $(date -u +%FT%TZ)"
rm -f /dev/shm/sm_segment.* 2>/dev/null
sleep 600
timeout 7200 $PY $TRAIN --tp 4 --layers 0 --steps 10 --warmup-steps 2 --trace --loss-log $O/${TAG}_full_tp4_real_trace.json > $O/${TAG}_full_tp4_real_trace.log 2>&1; echo "rc=$?"; grep -E "^step|layers," $O/${TAG}_full_tp4_real_trace.log | tail -12
echo "=== ${TAG} chain finished $(date -u +%FT%TZ)"
