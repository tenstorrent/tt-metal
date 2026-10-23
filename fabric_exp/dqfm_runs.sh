#!/bin/bash
# §64: flushed-signal benchmark (fabric_exp/streams/flush_mix.yaml: fused or standalone atomic signals, flush on/off,
# per packet and per chunk of 8/16) on three router versions, each a (fabric_erisc_router_v2.cpp,
# fabric_edm_packet_transmission.hpp) pair swapped in and restored (checked):
#   before = v2 before any atomic change (bits 26/28 as flags, run as 348673408), header at HEAD
#   s62    = flush-only staged atomic (set_state / flush / with_state)
#   s63    = deferred flushed atomic (the user's current files)
# Labels dqfm_<version>. Env: SRC_DIR (scratch dir holding the file sets).
cd /home/aagarwal/tt-metal
V2=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp
TR=tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp
run() {  # label flags
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 1500 fabric_exp/run_exp.sh "$1" fabric_exp/streams/flush_mix.yaml \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=$2 > /dev/null 2>&1
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT}"
    if [ -z "$res" ] || ! echo "$res" | grep -q ", 0 failed"; then pkill -x test_tt_fabric; sleep 2; tt-smi -r all > /dev/null 2>&1; echo "reset done"; fi
}
cp $V2 "$SRC_DIR/v2.orig_dqfm.cpp"; cp $TR "$SRC_DIR/transmission.orig_dqfm.hpp"
tt-smi -r all > /dev/null 2>&1
run dqfm_s63 13129088
cp "$SRC_DIR/fabric_erisc_router_v2.l.cpp" $V2; cp "$SRC_DIR/transmission.l.hpp" $TR
run dqfm_s62 13129088
cp "$SRC_DIR/fabric_erisc_router_v2.j.cpp" $V2; cp "$SRC_DIR/transmission.before.hpp" $TR
run dqfm_before 348673408
cp "$SRC_DIR/v2.orig_dqfm.cpp" $V2; cp "$SRC_DIR/transmission.orig_dqfm.hpp" $TR
cmp $V2 "$SRC_DIR/v2.orig_dqfm.cpp" && cmp $TR "$SRC_DIR/transmission.orig_dqfm.hpp" && echo "user files restored"
echo ALL_DONE
