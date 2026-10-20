#!/bin/bash
# DRISC offload (bit 21 = EXP_DRISC_OFFLOAD, P5: E1 pushes each slot into a DRISC ring; the DRISC decodes, forwards on
# NoC0 and delivers on NoC1) on top of 13129088 = 15226240. Slow dispatch throughout (baselines too, same mode).
# Validation, same-tree baselines (dqb_*), benchmarks, small packets, latency. Labels: dq_<flags>_*.
cd /home/aagarwal/tt-metal
KA="TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_METAL_SLOW_DISPATCH_MODE=1"
W="TT_FABRIC_EXP_WORKER_SLOTS=32 TT_FABRIC_EXP_MAX_SENDER_SLOTS=8"
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
run() {  # label yaml [env...]; returns 1 on failure or hang
    timeout 1500 fabric_exp/run_exp.sh "$1" "$2" TT_FABRIC_NO_BW_TELEMETRY=1 $KA "${@:3}" > /dev/null 2>&1
    local status=$?
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT (exit $status)} worker-bound=$(grep -c 'was worker-bound' fabric_exp/results/$1.log 2>/dev/null)"
    if [ $status -ne 0 ] || [ -z "$res" ] || ! echo "$res" | grep -q ", 0 failed"; then
        pkill -x test_tt_fabric; sleep 2; tt-smi -r all > /dev/null 2>&1
        return 1
    fi
    return 0
}
O=TT_FABRIC_EXP_FLAGS=15226240
B=TT_FABRIC_EXP_FLAGS=13129088
tt-smi -r all > /dev/null 2>&1
echo "== phase 1: validation"
run dq_15226240_mesh $SAN/test_fabric_mesh_sanity_common.yaml $O || { echo "RULED OUT: 15226240 (mesh sanity)"; exit 1; }
echo "== phase 2a: same-tree baselines (slow dispatch)"
run dqb_13129088_4352 fabric_exp/exp_2d_4352.yaml $B
run dqb_13129088_mix fabric_exp/streams/p3_mixes.yaml $B
run dqb_13129088_4352_trim fabric_exp/exp_2d_trim_4352.yaml $B
run dqb_13129088_mix_trim fabric_exp/streams/p3_mixes_trim.yaml $B
echo "== phase 2: benchmarks"
run dq_15226240_4352 fabric_exp/exp_2d_4352.yaml $O
run dq_15226240_mix fabric_exp/streams/p3_mixes.yaml $O
run dq_15226240_4352_trim fabric_exp/exp_2d_trim_4352.yaml $O
run dq_15226240_mix_trim fabric_exp/streams/p3_mixes_trim.yaml $O
run dq_15226240w32_4352 fabric_exp/exp_2d_4352.yaml $O $W
run dq_15226240w32_mix fabric_exp/streams/p3_mixes.yaml $O $W
echo "== phase 3: small packets and latency"
run dqb_13129088_small fabric_exp/exp_2d_trim_small.yaml $B
run dq_15226240_small fabric_exp/exp_2d_trim_small.yaml $O
run dqb_13129088_lat fabric_exp/exp_latency.yaml $B
run dq_15226240_lat fabric_exp/exp_latency.yaml $O
echo ALL_DONE
