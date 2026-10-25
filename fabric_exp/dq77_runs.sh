#!/bin/bash
# §77: the user's s77 on bh-lb-120-a08u24 (broker job: no resets, output streamed, heartbeat). s77 = the committed tree
# (s74c) plus: StaticSizedSenderEthChannel without the SlotAddresses table / buffer index (stride advance everywhere),
# compile-time free-slots stream ids in sender_channel_try_send and the sender steps, and no [[unlikely]] on the
# pending-atomic and TXQ-status branches. s77u = s77 with the five [[unlikely]] restored.
#   a: s77 validation (stream, mesh, 1D sanity) + 4352 / mix / flushmix / flushmix4 / 1D   labels dq77_13129088_<suite>
#   b: s74c (committed tree) and s77u, same benchmark suites                                labels dq74c_, dq77u_
#   c: repeats, s77 then s74c                                                               labels dq77r2_, dq74cr2_
#   d: s77u split: s77t = slot-table removal only (s74c router + s77 channels), s77s = compile-time stream ids only
#      (s77u router with the by-stride advance call kept, s74c channels); 4352 / mix, then s74c 4352 / mix
#                                                                                           labels dq77t_, dq77s_, dq74cr3_
#   e: s77s validation (stream, mesh, 1D sanity) + all benchmark suites; s77sn = s77s without the [[unlikely]]
#      (s77 router with the by-stride advance call kept, s74c channels), all benchmark suites
#                                                                                           labels dq77sv_, dq77sn_
# Flags 13129088, fast dispatch, telemetry off. Tree restored to s77 on exit, checked with cmp.
cd /home/aagarwal/tt-metal
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
R=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp
H=tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp
V=fabric_exp/variants
swap() { cp $V/$1/fabric_erisc_router_v2.cpp $R && cp $V/$1/fabric_erisc_datamover_channels.hpp $H && echo "== router: $1"; }
restore() { swap s77 > /dev/null
    cmp -s $V/s77/fabric_erisc_router_v2.cpp $R && cmp -s $V/s77/fabric_erisc_datamover_channels.hpp $H \
    && echo "tree restored to s77" || echo "!!! RESTORE FAILED"; }
save_elfs() {  # variant marker-file
    local n=0
    rm -rf $V/$1/elf
    for d in $(find /home/aagarwal/.tt-metal-cache/*/kernels/fabric_erisc_router_v2 -newer $2 -name '*.elf' \
        ! -name '*.xip.elf' -printf '%h\n' | sort -u); do
        local h=$(basename $(dirname $d)) k=$(basename $d) c=$(dirname $(dirname $d))
        mkdir -p $V/$1/elf/$h && cp $d/$k.elf $V/$1/elf/$h/ && cp $c/$h/named_ct_arg_map_generated.h $V/$1/elf/$h/ \
            && n=$((n+1))
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
            val1d) y=$SAN/test_fabric_sanity_at_least_2x2_mesh.yaml ;;
            4352) y=fabric_exp/exp_2d_4352.yaml ;; mix) y=fabric_exp/streams/p3_mixes.yaml ;;
            flushmix) y=fabric_exp/streams/flush_mix.yaml ;; flushmix4) y=fabric_exp/streams/flush_mix4.yaml ;;
            1d) y=fabric_exp/exp_1d_4352.yaml ;;
        esac
        run ${l}_13129088_$s $y || { echo STOPPED; exit 1; }
    done
}
B="4352 mix flushmix flushmix4 1d"
M=$(mktemp)
case "${1:-a}" in
a)  swap s77; sleep 1
    suites dq77 stream mesh val1d $B
    save_elfs s77 $M ;;
b)  swap s74c; sleep 1
    suites dq74c $B
    save_elfs s74c $M; touch $M; sleep 1
    swap s77u; sleep 1
    suites dq77u $B
    save_elfs s77u $M ;;
c)  swap s77; sleep 1
    suites dq77r2 $B
    swap s74c; sleep 1
    suites dq74cr2 $B ;;
d)  swap s77t; sleep 1
    suites dq77t 4352 mix
    save_elfs s77t $M; touch $M; sleep 1
    swap s77s; sleep 1
    suites dq77s 4352 mix
    save_elfs s77s $M
    swap s74c; sleep 1
    suites dq74cr3 4352 mix ;;
e)  swap s77s; sleep 1
    suites dq77sv stream mesh val1d $B
    touch $M; sleep 1
    swap s77sn; sleep 1
    suites dq77sn $B
    save_elfs s77sn $M ;;
esac
rm -f $M
echo ALL_DONE
