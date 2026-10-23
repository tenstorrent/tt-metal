#!/bin/bash
# §56 follow-up: which ERISC binds small E/W writes for base vs variant B (TX or RX +20 spin per packet), on
# fabric_exp/streams/p3_small_ew_ns.yaml. Also saves each variant's E0/E1 ELFs for the E/W sender/receiver routers.
# Labels dqpas_<variant>_<tx|rx|0>_13129088. Env: BASE_SRC, B_SRC, ORIG_BACKUP, ELF_DIR.
cd /home/aagarwal/tt-metal
F=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp
C=~/.cache/tt-metal-cache/8310703137207683420/kernels/fabric_erisc_router_v2
run() {
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 1500 fabric_exp/run_exp.sh "$1" fabric_exp/streams/p3_small_ew_ns.yaml \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 "${@:2}" > /dev/null 2>&1
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT}"
    if [ -z "$res" ] || ! echo "$res" | grep -q ", 0 failed"; then pkill -x test_tt_fabric; sleep 2; tt-smi -r all > /dev/null 2>&1; echo "reset done"; fi
}
save_elfs() {  # variant: copy every unprofiled no-delay 13129088 router ELF (E0 and E1) built by this variant's run
    mkdir -p "$ELF_DIR/$1"
    for d in $C/*/; do
        f=$d/named_ct_arg_map_generated.h
        grep -q '"EXP_FLAGS",13129088}' $f && grep -q '"PHASE_PROFILE_MODE",0}' $f && grep -q '"EXP_TX_DELAY",0}' $f && grep -q '"EXP_RX_DELAY",0}' $f || continue
        tag=$(tr '{' '\n' < $f | grep -E '"MY_DIRECTION"|"MY_ETH_CHANNEL"' | tr -dc '0-9,' )
        for e in $d/active_erisc/active_erisc.elf $d/subordinate_active_erisc/subordinate_active_erisc.elf; do
            [ -f $e ] && [ $e -nt "$ELF_DIR/.marker_$1" ] && cp $e "$ELF_DIR/$1/$(basename $(dirname $e))_dir${tag}.elf"
        done
    done
}
cp $F "$ORIG_BACKUP"
mkdir -p "$ELF_DIR"
tt-smi -r all > /dev/null 2>&1
for v in base B; do
    src=BASE_SRC; [ $v = B ] && src=B_SRC
    cp "${!src}" $F
    touch "$ELF_DIR/.marker_$v"
    run dqpas_${v}_0_13129088
    save_elfs $v
    run dqpas_${v}_tx_13129088 TT_FABRIC_EXP_TX_DELAY=20
    run dqpas_${v}_rx_13129088 TT_FABRIC_EXP_RX_DELAY=20
done
cp "$ORIG_BACKUP" $F
cmp $F "$ORIG_BACKUP" && echo "user v2 restored"
echo ALL_DONE
