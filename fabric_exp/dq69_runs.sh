#!/bin/bash
# §70: FLUSH_RELAXED opt-in (s69) on bh-lb-120-a08u24 (broker job: no resets, output streamed, heartbeat). Header and test
# kernel stay at s69 throughout; router + transmission are swapped, and the test kernel's kActiveFlushMode toggled:
#   s69  RELAXED   validation + 4352 / mix / latency / flushed-signal        labels dq69_13129088_<suite>
#   s69  STRICT    mix / flushed-signal (new code, strict packets)           labels dq69s_13129088_<suite>
#   s63g           4352 / mix / flushed-signal (RELAXED treated as STRICT)   labels a8s63g2_13129088_<suite>
#   s69  RELAXED   repeat 4352 / mix / flushed-signal                        labels dq69r2_13129088_<suite>
# Phase c: s69a (no completion guard, bypass bounded by the trid window), validation + 4352 / mix / flushed-signal x2,
# labels dq69a / dq69ar2. Phase a = first two lines, phase b = last two (broker jobs are capped at 25 min). Flags 13129088, fast dispatch, telemetry off. Tree restored to s69 (RELAXED) on exit, checked with cmp.
cd /home/aagarwal/tt-metal
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
R=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp
T=tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp
K=tests/tt_metal/tt_fabric/test_infra/kernels/tt_fabric_test_kernels_utils.hpp
V=fabric_exp/variants
C=/home/aagarwal/.tt-metal-cache/tt-metal-cache8310703137207683420/kernels/fabric_erisc_router_v2
swap() { cp $V/$1/fabric_erisc_router_v2.cpp $R && cp $V/$1/fabric_edm_packet_transmission.hpp $T && echo "== router: $1"; }
mode() { cp $V/s69/tt_fabric_test_kernels_utils.hpp $K
    [ $1 = STRICT ] && sed -i 's/kActiveFlushMode = tt::tt_fabric::NocFlushType::FLUSH_RELAXED/kActiveFlushMode = tt::tt_fabric::NocFlushType::FLUSH_STRICT/' $K
    echo "== test kernel: $(grep -o 'kActiveFlushMode = [A-Za-z_:]*' $K)"; }
restore() { swap s69 > /dev/null; mode RELAXED > /dev/null
    cmp -s $V/s69/fabric_erisc_router_v2.cpp $R && cmp -s $V/s69/fabric_edm_packet_transmission.hpp $T && cmp -s $V/s69/tt_fabric_test_kernels_utils.hpp $K \
    && echo "tree restored to s69" || echo "!!! RESTORE FAILED"; }
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
case "${1:-a}" in
a)  swap s69; mode RELAXED; sleep 1
    suites dq69 stream mesh 4352 mix lat flushmix
    save_elfs s69 $M
    mode STRICT
    suites dq69s mix flushmix ;;
b)  swap s63g; mode RELAXED
    suites a8s63g2 4352 mix flushmix
    swap s69
    suites dq69r2 4352 mix flushmix ;;
c)  swap s69a; mode RELAXED; sleep 1
    suites dq69a stream mesh 4352 mix flushmix
    save_elfs s69a $M
    suites dq69ar2 flushmix ;;
esac
rm -f $M
echo ALL_DONE
