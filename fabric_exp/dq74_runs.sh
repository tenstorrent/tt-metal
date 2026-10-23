#!/bin/bash
# §74: the user's out-of-order-safe two-slot check (s74) on bh-lb-120-a08u24 (broker job: no resets, output streamed,
# heartbeat). At the first bypass, global outstanding, trid_s and trid_p are read in one group; g == s + p proves every
# other trid is idle; the pending state waits on whichever of s / p is non-zero. Standalone atomic: last write trid =
# transaction_id - 1. Bypass bound > 1. can_defer with >= 16 receiver slots. Header, api_common.h and test kernel
# (RELAXED) are s71's throughout.
#   a: s74 validation (stream, mesh) + 4352 / mix / flushmix / flushmix4   labels dq74_13129088_<suite>
#   b: s74 repeat 4352 / mix / flushmix / flushmix4                        labels dq74r2_13129088_<suite>
#      s71b same-day control flushmix / flushmix4                          labels dq71br3_13129088_<suite>
# Flags 13129088, fast dispatch, telemetry off. Tree restored to s74 on exit, checked with cmp.
cd /home/aagarwal/tt-metal
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
R=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp
T=tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp
V=fabric_exp/variants
C=/home/aagarwal/.tt-metal-cache/tt-metal-cache8310703137207683420/kernels/fabric_erisc_router_v2
swap() { cp $V/$1/fabric_erisc_router_v2.cpp $R && cp $V/$1/fabric_edm_packet_transmission.hpp $T && echo "== router: $1"; }
restore() { swap s74 > /dev/null
    cmp -s $V/s74/fabric_erisc_router_v2.cpp $R && cmp -s $V/s74/fabric_edm_packet_transmission.hpp $T \
    && echo "tree restored to s74" || echo "!!! RESTORE FAILED"; }
save_elfs() {  # variant marker-file
    local n=0
    rm -rf $V/$1/elf
    for d in $(find $C -newer $2 -name '*.elf' ! -name '*.xip.elf' -printf '%h\n' | sort -u); do
        local h=$(basename $(dirname $d)) k=$(basename $d)
        mkdir -p $V/$1/elf/$h && cp $d/$k.elf $V/$1/elf/$h/ && cp $C/$h/named_ct_arg_map_generated.h $V/$1/elf/$h/ && n=$((n+1))
    done
    echo "saved $n ELFs for $1"
}
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null; restore' EXIT
run() {  # label yaml
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 1200 fabric_exp/run_exp.sh "$1" "$2" \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 \
        | grep --line-buffered -v 'Running Test'
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT}"
    [ -n "$res" ] && echo "$res" | grep -q ", 0 failed"
}
suites() {  # label-prefix suite...
    local l=$1; shift
    for s in "$@"; do
        case $s in
            stream) y=$SAN/test_fabric_stream_sanity.yaml ;; mesh) y=$SAN/test_fabric_mesh_sanity_common.yaml ;;
            4352) y=fabric_exp/exp_2d_4352.yaml ;; mix) y=fabric_exp/streams/p3_mixes.yaml ;;
            flushmix) y=fabric_exp/streams/flush_mix.yaml ;; flushmix4) y=fabric_exp/streams/flush_mix4.yaml ;;
        esac
        run ${l}_13129088_$s $y || { echo STOPPED; exit 1; }
    done
}
M=$(mktemp)
case "${1:-a}" in
a)  swap s74; sleep 1
    suites dq74 stream mesh 4352 mix flushmix flushmix4
    save_elfs s74 $M ;;
b)  swap s74; sleep 1
    suites dq74r2 4352 mix flushmix flushmix4
    swap s71b; sleep 1
    suites dq71br3 flushmix flushmix4 ;;
esac
rm -f $M
echo ALL_DONE
