#!/bin/bash
# §79: the user's s79 on bh-lb-120-a08u24 (broker job: no resets, output streamed, heartbeat). s79 = s77s plus:
# the worker connection transitions moved into noinline establish_worker_connection_cold /
# teardown_worker_connection_cold helpers with the disconnected poll kept inline (s79c = this part alone), and the
# grouped free-slots mask narrowed from the 17-bit field mask to 0x7FF, which fits an andi immediate (s79m = this part
# alone).
#   s79 validation (stream, mesh, 1D sanity) + 4352 / mix / flushmix / flushmix4 / 1D        label dq79_
#   s79c, s79m, s77s (control) benchmark suites; then an s79 repeat         labels dq79c_, dq79m_, dq77s4_, dq79r2_
#   (phase a; phase b repeats s79m and s79c: labels dq79mr2_, dq79cr2_)
# Every variant's new router ELFs get the stack check after its first suite: the three largest stack frames plus 256 B
# for the firmware must fit in the local memory left above .bss (top of local memory 0xFFB02000).
# Flags 13129088, fast dispatch, telemetry off. Tree restored to s79 on exit, checked with cmp.
cd /home/aagarwal/tt-metal
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
R=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp
H=tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp
P=tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp
V=fabric_exp/variants
OD=runtime/sfpi/compiler/bin/riscv-tt-elf-objdump
NM=runtime/sfpi/compiler/bin/riscv-tt-elf-nm
swap() { cp $V/$1/fabric_erisc_router_v2.cpp $R && cp $V/$1/fabric_erisc_datamover_channels.hpp $H \
    && cp $V/$1/fabric_erisc_router_speedy_path.hpp $P && echo "== router: $1"; }
restore() { swap s79 > /dev/null
    cmp -s $V/s79/fabric_erisc_router_v2.cpp $R && cmp -s $V/s79/fabric_erisc_datamover_channels.hpp $H \
    && cmp -s $V/s79/fabric_erisc_router_speedy_path.hpp $P && echo "tree restored to s79" || echo "!!! RESTORE FAILED"; }
new_elfs() {  # marker-file
    find /home/aagarwal/.tt-metal-cache/*/kernels/fabric_erisc_router_v2 -newer $1 -name '*.elf' ! -name '*.xip.elf'
}
save_elfs() {  # variant marker-file
    local n=0
    rm -rf $V/$1/elf
    for d in $(new_elfs $2 | xargs -r -n1 dirname | sort -u); do
        local h=$(basename $(dirname $d)) k=$(basename $d) c=$(dirname $(dirname $d))
        mkdir -p $V/$1/elf/$h && cp $d/$k.elf $V/$1/elf/$h/ && cp $c/$h/named_ct_arg_map_generated.h $V/$1/elf/$h/ \
            && n=$((n+1))
    done
    echo "saved $n ELFs for $1"
}
stack_check() {  # marker-file
    local min=99999 n=0
    for e in $(new_elfs $1); do
        local f=$($OD -d $e | grep -oE 'addi\s+sp,sp,-[0-9]+' | grep -oE '[0-9]+$' | sort -rn | head -3 | paste -sd+ | bc)
        local free=$(( 0xFFB02000 - 0x$($NM $e | awk '$3=="__ldm_bss_end"{print $1}') ))
        local m=$((free - f - 256)); n=$((n+1))
        [ $m -lt $min ] && min=$m && echo "   $(basename $e) $(basename $(dirname $(dirname $e))): frames $f, free $free"
    done
    echo "stack check: $n ELFs, smallest margin $min B"
    [ $min -ge 0 ] || { echo STOPPED; exit 1; }
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
variant() {  # variant label-prefix suite...
    local v=$1 l=$2; shift 2
    touch $M; sleep 1
    swap $v; sleep 1
    suites $l $1; stack_check $M; shift
    suites $l "$@"
    [ -d $V/$v/elf ] && [ "$KEEP_ELFS" = 1 ] || save_elfs $v $M
}
case "${1:-a}" in
a)  variant s79 dq79 stream mesh val1d $B
    variant s79c dq79c $B
    variant s79m dq79m $B
    KEEP_ELFS=1 variant s77s dq77s4 $B
    KEEP_ELFS=1 variant s79 dq79r2 $B ;;
b)  KEEP_ELFS=1 variant s79m dq79mr2 $B
    KEEP_ELFS=1 variant s79c dq79cr2 $B ;;
esac
rm -f $M
echo ALL_DONE
