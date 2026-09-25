#!/bin/bash
# E8: packed [deep hot, cold] vs the same segments alone on the original path, real history, S0 layers.
cd "$(dirname "$0")"; D=results/phaseb_e8
until ! pgrep -f "batch_e4[.]sh" >/dev/null; do sleep 20; done
G="E8=141312:2048,0:2048"
common="HARNESS=budget_packed.py EXP=E8 LAYER_SET=S0 BUDGET_LAYER_IDS=0,1,2,3,4,5,6,7 BUDGET_COMPOS=$G BUDGET_CAPACITY=143360"
env $common RUN_ID=e8_ref BUDGET_REFERENCE=1 BUDGET_DUMP_KV=$PWD/$D/ref ./run_budget.sh
env $common RUN_ID=e8_packed BUDGET_B=2 BUDGET_DUMP_KV=$PWD/$D/packed ./run_budget.sh
../python_env/bin/python3 compare_kv.py $D/ref $D/packed 0:143360 1:2048 | tee results/logs/e8_compare.txt
