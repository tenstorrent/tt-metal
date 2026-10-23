#!/bin/bash
# §67 split on bh-lb-120-a08u24 (broker job), one change at a time, each on 4352 / mix / flushed-signal with ELFs saved:
#   s67a  §67 with one drain before the forward on every router (no E/W post-forward drain)   labels dq67a_13129088_<suite>
#   s67b  §67 with the idle drain back before the completion section (§63's place)           labels dq67b_13129088_<suite>
#   s67c  §63+guard router with §67's transmission header (fused-only deferral, no landed     labels dq67c_13129088_<suite>
#         check; atomic and scatter flush inline)
# Flags 13129088, fast dispatch, telemetry off. The tree is restored to s67 on exit, checked with cmp.
cd /home/aagarwal/tt-metal
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
R=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp
T=tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp
V=fabric_exp/variants
C=/home/aagarwal/.tt-metal-cache/tt-metal-cache8310703137207683420/kernels/fabric_erisc_router_v2
swap() { cp $V/$1/fabric_erisc_router_v2.cpp $R && cp $V/$1/fabric_edm_packet_transmission.hpp $T && echo "== router: $1"; }
restore() { swap s67 > /dev/null; cmp -s $V/s67/fabric_erisc_router_v2.cpp $R && cmp -s $V/s67/fabric_edm_packet_transmission.hpp $T \
    && echo "tree restored to s67" || echo "!!! RESTORE FAILED"; }
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
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088
    local status=$?
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT (exit $status)}"
    [ $status -eq 0 ] && [ -n "$res" ] && echo "$res" | grep -q ", 0 failed"
}
suites() {  # label-prefix suite...
    local l=$1; shift
    for s in "$@"; do
        case $s in
            stream) y=$SAN/test_fabric_stream_sanity.yaml ;; mesh) y=$SAN/test_fabric_mesh_sanity_common.yaml ;;
            4352) y=fabric_exp/exp_2d_4352.yaml ;; mix) y=fabric_exp/streams/p3_mixes.yaml ;;
            lat) y=fabric_exp/exp_latency.yaml ;; flushmix) y=fabric_exp/streams/flush_mix.yaml ;;
        esac
        run ${l}_13129088_$s $y || { echo STOPPED; exit 1; }
    done
}
M=$(mktemp)
for pair in s67a:dq67a s67b:dq67b s67c:dq67c; do
    v=${pair%%:*} l=${pair##*:}
    swap $v; touch $M; sleep 1
    suites $l 4352 mix flushmix
    save_elfs $v $M
done
rm -f $M
echo ALL_DONE
