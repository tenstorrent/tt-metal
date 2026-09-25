#!/bin/bash
# E8 control: the deep segment packed with a different cold companion (stream 7), same packed history fill.
cd "$(dirname "$0")"; D=$PWD/results/phaseb_e8
until ! pgrep -f "batch_e7[.]sh" >/dev/null; do sleep 30; done
HARNESS=budget_packed.py RUN_ID=e8t_packed_ctl EXP=E8T LAYER_SET=S0 BUDGET_LAYER_IDS=0,1,2,3,4,5,6,7 \
  BUDGET_COMPOS="E8c=141312:2048,7@0:2048" BUDGET_CAPACITY=143360 BUDGET_B=2 BUDGET_TOPK_DUMP=$D/topk_packed_ctl.pt ./run_budget.sh
# packed vs packed-with-another-companion: both are "packed" captures, so compare them as ref/packed with
# the same (layer-major) order by treating the control as a packed capture.
../python_env/bin/python3 - "$D/topk_packed.pt" "$D/topk_packed_ctl.pt" <<'PY' | tee results/logs/e8_topk_control.txt
import sys, torch
A, B = torch.load(sys.argv[1]), torch.load(sys.argv[2])
S = len(A["segments"]); L = len(A["ids"]) // S
for l in range(L):
    a, b = A["ids"][l * S].long(), B["ids"][l * S].long()  # segment 0 = the deep hot segment
    o = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).sum(-1).float() / a.shape[-1]
    print(f"deep segment, sparse layer {l}: mean overlap {float(o.mean()):.4f} rows identical {float((o == 1).float().mean()):.3f}")
PY
