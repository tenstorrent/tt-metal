#!/bin/bash
# Per-layer heterogeneity: are sparse layers 3-7 cheaper than 8-15? Same 5-layer count, cold + deep.
cd "$(dirname "$0")"
until ! pgrep -f "batch_e3[.]sh" >/dev/null; do sleep 20; done
for spec in "s3_7:3,4,5,6,7" "s8_12:8,9,10,11,12" "s11_15:11,12,13,14,15"; do
  name=${spec%%:*}; ids=${spec#*:}
  RUN_ID=lh_${name} EXP=LH LAYER_SET=${name^^} BUDGET_LAYER_IDS=$ids BUDGET_W=2048 \
    BUDGET_POINTS=0:2048,141312:2048,548864:2048 ./run_budget.sh
  RUN_ID=lh_${name}_w8192 EXP=LH LAYER_SET=${name^^} BUDGET_LAYER_IDS=$ids BUDGET_W=8192 BUDGET_POINTS=0:8192 ./run_budget.sh
done
