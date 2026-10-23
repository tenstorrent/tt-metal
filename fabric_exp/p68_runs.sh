#!/bin/bash
# §68: where does deferral's +2 cycles/packet go on rows that never signal? §62 behavior (variants/s62x) against
# §63 + guard (variants/s63g) on streams/p68_profile.yaml (1-hop E/W write / scatter / unflushed fused, forwarded 16 B
# write / scatter). Per variant: no profiling; E1 slack (TT_FABRIC_EXP_RX_DELAY 20, 40); level-2 phase profile, one
# phase per run (TT_FABRIC_PHASE_MASK, no per-iteration timing): decode/admit 16, forward 32, local 64, completion 128,
# whole receiver step 2048. Broker job: no resets, line-buffered output, heartbeat. Labels p68_<variant>_<run>.
# Flags 13129088, fast dispatch, telemetry off. The tree is restored to s63g on exit, checked with cmp.
cd /home/aagarwal/tt-metal
R=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp
T=tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp
V=fabric_exp/variants
Y=fabric_exp/streams/p68_profile.yaml
swap() { cp $V/$1/fabric_erisc_router_v2.cpp $R && cp $V/$1/fabric_edm_packet_transmission.hpp $T && echo "== router: $1"; }
restore() { swap s63g > /dev/null; cmp -s $V/s63g/fabric_erisc_router_v2.cpp $R && cmp -s $V/s63g/fabric_edm_packet_transmission.hpp $T \
    && echo "tree restored to s63g" || echo "!!! RESTORE FAILED"; }
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null; restore' EXIT
run() {  # label [ENV=VALUE ...]
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 600 fabric_exp/run_exp.sh "$1" $Y \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 "${@:2}" \
        | grep --line-buffered -v 'Running Test'
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT}"
    [ -n "$res" ] && echo "$res" | grep -q ", 0 failed"
}
for v in s62x s63g; do
    swap $v
    run p68_${v}_noprof || { echo STOPPED; exit 1; }
    for d in 20 40; do run p68_${v}_rxd$d TT_FABRIC_EXP_RX_DELAY=$d || { echo STOPPED; exit 1; }; done
    for m in 16 32 64 128 2048; do
        run p68_${v}_m$m TT_FABRIC_PHASE_PROFILE=2 TT_FABRIC_PHASE_MASK=$m TT_FABRIC_PHASE_NO_ITER=1 || { echo STOPPED; exit 1; }
    done
done
echo ALL_DONE
