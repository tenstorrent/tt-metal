#!/bin/bash
# §55: where E1's per-packet time goes on 1-hop writes (v2 with E1 bits 27/28 hard-wired, flags 13129088, fast
# dispatch, fabric_exp/exp_2d_4352_1hop_write.yaml). Each run times ONE phase (TT_FABRIC_PHASE_PROFILE=2 with
# TT_FABRIC_PHASE_MASK, no per-iteration timing); four extra E1 segments come from a temporary instrumented copy of
# v2 (VARIANT_SRC) on phase ids E0 never hits in this traffic: RX_ACK = packet poll, TX_TXQ_PRESPIN = admitted ->
# local write, TX_CONN = local write done -> packet retired, TX_TXQ_SPIN = bit-22 completion block. The user's file
# is restored afterwards (checked). Labels dqe1s_m<mask>_13129088_1hop, plus dqe1s_noprof_13129088_1hop.
cd /home/aagarwal/tt-metal
F=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp
Y=fabric_exp/exp_2d_4352_1hop_write.yaml
run() {
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 1500 fabric_exp/run_exp.sh "$1" $Y \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 "${@:2}" > /dev/null 2>&1
    local status=$?
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT (exit $status)}"
    if [ $status -ne 0 ] || [ -z "$res" ] || ! echo "$res" | grep -q ", 0 failed"; then
        pkill -x test_tt_fabric; sleep 2; tt-smi -r all > /dev/null 2>&1; echo "reset done"
    fi
}
cp $F "$ORIG_BACKUP"
tt-smi -r all > /dev/null 2>&1
run dqe1s_noprof_13129088_1hop
cp "$VARIANT_SRC" $F
for m in 256 16 512 64 8 2 128 2048; do
    run dqe1s_m${m}_13129088_1hop TT_FABRIC_PHASE_PROFILE=2 TT_FABRIC_PHASE_MASK=$m TT_FABRIC_PHASE_NO_ITER=1
done
cp "$ORIG_BACKUP" $F
cmp $F "$ORIG_BACKUP" && echo "user v2 restored"
echo ALL_DONE
