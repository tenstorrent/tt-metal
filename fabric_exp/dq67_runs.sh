#!/bin/bash
# §67 on bh-lb-120-a08u24 (broker job: no resets, output streamed, heartbeat). Router variants are swapped in, run, and
# their ERISC ELFs saved to variants/<v>/elf/ (the JIT rebuilds each config in place):
#   s67   the user's §67 (validation + 4352 / mix / latency / flushed-signal)      labels dq67_13129088_<suite>
#   s62x  §63+guard with can_defer_local_atomic_flush() forced false = §62 behavior labels a8s62x_13129088_<suite>
#   s67   repeat on 4352 / mix / flushed-signal                                     labels dq67r2_13129088_<suite>
# §63+guard (s63g) results are a8s63 / a8s63b / a8s63g. Flags 13129088, fast dispatch, telemetry off.
# The tree is restored to s67 on exit, checked with cmp.
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
swap s67; M=$(mktemp); sleep 1
suites dq67 stream mesh 4352 mix lat flushmix
save_elfs s67 $M
swap s62x; touch $M; sleep 1
suites a8s62x 4352 mix lat flushmix
save_elfs s62x $M
swap s67
suites dq67r2 4352 mix flushmix
rm -f $M
echo ALL_DONE
