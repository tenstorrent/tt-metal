#!/bin/bash
# Noise check for bits 24/25: two more untrimmed 4352 B runs and east-west mix runs per main config (labels r<N>_*).
cd /home/aagarwal/tt-metal
KA="TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000"
W="TT_FABRIC_EXP_WORKER_SLOTS=32 TT_FABRIC_EXP_MAX_SENDER_SLOTS=8"
run() {
    timeout 1500 fabric_exp/run_exp.sh "$1" "$2" TT_FABRIC_NO_BW_TELEMETRY=1 $KA "${@:3}" > /dev/null 2>&1 || { pkill -x test_tt_fabric; sleep 2; tt-smi -r all > /dev/null 2>&1; }
    echo "$1: $(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)"
}
for rep in 1 2; do
    for f in 13129088 29906304 46683520 63460736; do
        run r${rep}_${f}_4352 fabric_exp/exp_2d_4352.yaml TT_FABRIC_EXP_FLAGS=$f
        run r${rep}_${f}_mixew fabric_exp/streams/p3_mixes_ew.yaml TT_FABRIC_EXP_FLAGS=$f
    done
    for f in 15226240 32003456 65557888; do
        run r${rep}_${f}w32_4352 fabric_exp/exp_2d_4352.yaml TT_FABRIC_EXP_FLAGS=$f $W
    done
done
echo ALL_DONE
