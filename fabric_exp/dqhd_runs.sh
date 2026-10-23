#!/bin/bash
# §57: the user's v2 with variant A folded in, E1 bit 26 (variant B: pre-admission word-40 load reused by local delivery)
# and E1 bit 28 (hold 1..3-packet completion tails for 4 consecutive idle polls). ERISC-only, fast dispatch.
# Per flags value: mesh validation, 4352, mix, latency, and a TX-queue-counter run of the small E/W and N/S writes
# (completion frames per packet). Labels dqhd_<flags>_<suite>. A failing variant is reset and skipped.
cd /home/aagarwal/tt-metal
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
run() {  # label yaml flags [env...]
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 1500 fabric_exp/run_exp.sh "$1" "$2" \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=$3 "${@:4}" > /dev/null 2>&1
    local status=$?
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT (exit $status)} worker-bound=$(grep -c 'was worker-bound' fabric_exp/results/$1.log 2>/dev/null)"
    if [ $status -ne 0 ] || [ -z "$res" ] || ! echo "$res" | grep -q ", 0 failed"; then
        pkill -x test_tt_fabric; sleep 2; tt-smi -r all > /dev/null 2>&1; echo "reset done"; return 1
    fi
}
variant() {
    local f=$1
    run dqhd_${f}_mesh $SAN/test_fabric_mesh_sanity_common.yaml $f || return 0
    run dqhd_${f}_4352 fabric_exp/exp_2d_4352.yaml $f || return 0
    run dqhd_${f}_mix fabric_exp/streams/p3_mixes.yaml $f || return 0
    run dqhd_${f}_lat fabric_exp/exp_latency.yaml $f || return 0
    run dqhd_${f}_txq fabric_exp/streams/p3_small_ew_ns.yaml $f TT_FABRIC_TXQ_COUNTERS=1 || return 0
}
tt-smi -r all > /dev/null 2>&1
for f in ${DQHD_VARIANTS:-348673408 13129088 80237952 281564544}; do variant $f; done
echo ALL_DONE
