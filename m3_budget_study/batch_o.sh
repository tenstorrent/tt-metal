#!/bin/bash
# Stage overhead o(W, h) = 2*T(1 layer) - T(2 layers), for sparse layer 8 (+9) and dense layer 0 (+1).
cd "$(dirname "$0")"
until ! pgrep -f "batch_e3b[.]sh" >/dev/null; do sleep 20; done
for spec in "o_s1:8" "o_s2:8,9" "o_d1:0" "o_d2:0,1"; do
  name=${spec%%:*}; ids=${spec#*:}
  for W in 2048 8192; do
    RUN_ID=${name}_w$W EXP=O LAYER_SET=${name^^} BUDGET_LAYER_IDS=$ids BUDGET_W=$W \
      BUDGET_POINTS=0:$W,139264:$W,548864:$W ./run_budget.sh
  done
done
