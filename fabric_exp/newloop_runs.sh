#!/bin/bash
# §29's 4352 B runs (same settings as slot_runs.sh) with the new benchmark send loop, labels n_*.
cd /home/aagarwal/tt-metal
KA="TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000"
W="TT_FABRIC_EXP_WORKER_SLOTS=32 TT_FABRIC_EXP_MAX_SENDER_SLOTS=8"
run() {  # label yaml [env...]
    timeout 900 fabric_exp/run_exp.sh "$1" "$2" TT_FABRIC_NO_BW_TELEMETRY=1 "${@:3}" > /dev/null 2>&1
    if [ ! -f "fabric_exp/results/$1.csv" ]; then echo "!!! STOPPING: $1 failed"; pkill -x test_tt_fabric; tt-smi -r all > /dev/null 2>&1; exit 1; fi
    echo "$1: $(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log) worker-bound=$(grep -c 'was worker-bound' fabric_exp/results/$1.log)"
}
tt-smi -r all > /dev/null 2>&1
for t in trim_ ""; do
    Y=fabric_exp/exp_2d_${t}4352.yaml; L=4352${t:+_trim}
    run n_stockka8000_$L $Y
    run n_stock_$L $Y $KA
    run n_546176_$L $Y $KA TT_FABRIC_EXP_FLAGS=546176
    run n_4740480_$L $Y $KA TT_FABRIC_EXP_FLAGS=4740480
    run n_13129088_$L $Y $KA TT_FABRIC_EXP_FLAGS=13129088
    run n_13129088w32_$L $Y $KA TT_FABRIC_EXP_FLAGS=13129088 $W
    run n_15226240w32_$L $Y $KA TT_FABRIC_EXP_FLAGS=15226240 $W
done
echo ALL_DONE
