#!/bin/bash
# §47: user's router kernel v2 (fabric_erisc_router_v2.cpp, TT_FABRIC_KERNEL_VERSION=2) against v1 (asm restored).
# First a v1 check (labels dqv1*), then the full fast-dispatch suite on v2 (labels dqk / dqkb).
cd /home/aagarwal/tt-metal
run() {
    env -u TT_METAL_SLOW_DISPATCH_MODE timeout 900 fabric_exp/run_exp.sh "$1" "$2" TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 "${@:3}" > /dev/null 2>&1
    local s=$? r=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log)
    echo "$1: ${r:-NO RESULT (exit $s)}"
}
tt-smi -r all > /dev/null 2>&1
echo "== v1 check"
run dqv1b_13129088_4352 fabric_exp/exp_2d_4352.yaml TT_FABRIC_EXP_FLAGS=13129088
run dqv1b_13129088_mix fabric_exp/streams/p3_mixes.yaml TT_FABRIC_EXP_FLAGS=13129088
run dqv1_15226240_4352 fabric_exp/exp_2d_4352.yaml TT_FABRIC_EXP_FLAGS=15226240
run dqv1_15226240_mix fabric_exp/streams/p3_mixes.yaml TT_FABRIC_EXP_FLAGS=15226240
echo "== v2 suite"
TT_FABRIC_KERNEL_VERSION=2 bash fabric_exp/dqf_runs.sh dqk dqkb
