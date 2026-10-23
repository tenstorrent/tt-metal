#!/bin/bash
# §56: E1 local-delivery header loads (v2 with E1 bits 27/28 hard-wired, flags 13129088, fast dispatch). Variants are
# whole copies of the router file swapped in and restored (checked): base = the user's file, A = local delivery takes
# the payload size from its packed word-40 load (caller's lhu dead), B = A + the pre-admission src_ch_id byte load
# widened to word 40 and reused by 2D local delivery. Per variant: mesh validation, 4352, mix, latency.
# Labels dqpa_<variant>_13129088_<suite>. Env: BASE_SRC, A_SRC, B_SRC, ORIG_BACKUP.
cd /home/aagarwal/tt-metal
F=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
run() {  # label yaml; returns 1 on failure
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 1500 fabric_exp/run_exp.sh "$1" "$2" \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 > /dev/null 2>&1
    local status=$?
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT (exit $status)} worker-bound=$(grep -c 'was worker-bound' fabric_exp/results/$1.log 2>/dev/null)"
    if [ $status -ne 0 ] || [ -z "$res" ] || ! echo "$res" | grep -q ", 0 failed"; then
        pkill -x test_tt_fabric; sleep 2; tt-smi -r all > /dev/null 2>&1; echo "reset done"; return 1
    fi
}
variant() {  # name src
    cp "$2" $F
    run dqpa_$1_13129088_mesh $SAN/test_fabric_mesh_sanity_common.yaml || return 0
    run dqpa_$1_13129088_4352 fabric_exp/exp_2d_4352.yaml || return 0
    run dqpa_$1_13129088_mix fabric_exp/streams/p3_mixes.yaml || return 0
    run dqpa_$1_13129088_lat fabric_exp/exp_latency.yaml || return 0
}
cp $F "$ORIG_BACKUP"
tt-smi -r all > /dev/null 2>&1
variant B "$B_SRC"
variant A "$A_SRC"
variant base "$BASE_SRC"
cp "$ORIG_BACKUP" $F
cmp $F "$ORIG_BACKUP" && echo "user v2 restored"
echo ALL_DONE
