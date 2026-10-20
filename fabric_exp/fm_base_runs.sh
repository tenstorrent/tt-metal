#!/bin/bash
# Same-tree baselines for fm_runs.sh: the bit 26-28 change zeroes two new globals unconditionally, so the 13129088 and
# 15226240 builds may differ from the ak_* builds. Labels: fmb_<flags>_*.
cd /home/aagarwal/tt-metal
KA="TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000"
W="TT_FABRIC_EXP_WORKER_SLOTS=32 TT_FABRIC_EXP_MAX_SENDER_SLOTS=8"
run() {
    timeout 1500 fabric_exp/run_exp.sh "$1" "$2" TT_FABRIC_NO_BW_TELEMETRY=1 $KA "${@:3}" > /dev/null 2>&1
    local status=$?
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT (exit $status)} worker-bound=$(grep -c 'was worker-bound' fabric_exp/results/$1.log 2>/dev/null)"
    if [ $status -ne 0 ] || [ -z "$res" ] || ! echo "$res" | grep -q ", 0 failed"; then
        pkill -x test_tt_fabric; sleep 2; tt-smi -r all > /dev/null 2>&1
    fi
}
run fmb_13129088_mix fabric_exp/streams/p3_mixes.yaml TT_FABRIC_EXP_FLAGS=13129088
run fmb_13129088_4352 fabric_exp/exp_2d_4352.yaml TT_FABRIC_EXP_FLAGS=13129088
run fmb_13129088_mix_trim fabric_exp/streams/p3_mixes_trim.yaml TT_FABRIC_EXP_FLAGS=13129088
run fmb_13129088_4352_trim fabric_exp/exp_2d_trim_4352.yaml TT_FABRIC_EXP_FLAGS=13129088
run fmb_15226240w32_4352 fabric_exp/exp_2d_4352.yaml TT_FABRIC_EXP_FLAGS=15226240 $W
run fmb_15226240w32_mix fabric_exp/streams/p3_mixes.yaml TT_FABRIC_EXP_FLAGS=15226240 $W
echo ALL_DONE
