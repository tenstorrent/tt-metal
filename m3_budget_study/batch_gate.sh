#!/bin/bash
# Phase B correctness gate: packed (prefill_segments) vs the original prefill_chunk path, KV PCC.
cd "$(dirname "$0")"; D=results/phaseb_gate
G="G1=0:2048,0:2048;G2=6144:2048,1@8192:1024;G3=10240:256,5@12288:2048"
common="HARNESS=budget_packed.py EXP=GATE LAYER_SET=L0_3 BUDGET_LAYER_IDS=0,3 BUDGET_COMPOS=$G BUDGET_CAPACITY=14336"
env $common RUN_ID=gate_ref BUDGET_REFERENCE=1 BUDGET_DUMP_KV=$PWD/$D/ref ./run_budget.sh
env $common RUN_ID=gate_packed BUDGET_B=2 BUDGET_DUMP_KV=$PWD/$D/packed ./run_budget.sh
../python_env/bin/python3 compare_kv.py $D/ref $D/packed 0:10496 1:14336 | tee results/logs/gate_compare.txt
