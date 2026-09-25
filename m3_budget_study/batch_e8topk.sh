#!/bin/bash
# E8 indexer top-k overlap: packed [deep hot, cold] vs the same segments alone, S0 layers, real history.
cd "$(dirname "$0")"; D=$PWD/results/phaseb_e8
until ! pgrep -f "batch_e6[.]sh" >/dev/null; do sleep 20; done
G="E8=141312:2048,0:2048"
common="HARNESS=budget_packed.py EXP=E8T LAYER_SET=S0 BUDGET_LAYER_IDS=0,1,2,3,4,5,6,7 BUDGET_COMPOS=$G BUDGET_CAPACITY=143360"
env $common RUN_ID=e8t_ref BUDGET_REFERENCE=1 BUDGET_TOPK_DUMP=$D/topk_ref.pt ./run_budget.sh
env $common RUN_ID=e8t_packed BUDGET_B=2 BUDGET_TOPK_DUMP=$D/topk_packed.pt ./run_budget.sh
../python_env/bin/python3 topk_overlap.py $D/topk_ref.pt $D/topk_packed.pt | tee results/logs/e8_topk_overlap.txt
