#!/bin/bash
# Phase 0, job B, on bh-lb-120-a08u24 (broker job: no resets, output streamed, heartbeat). How much s79mc's numbers
# move when nothing about the router's work changes:
#   -O2, -Os (production's level when VC1 is active)   validation + 4352 / mix / flushmix / flushmix4 / 1D
#     each in its own fresh kernel cache (with a warm cache TT_METAL_FABRIC_OPT_LEVEL reuses the -O3 ELFs: job A),
#     compile commands logged for offline builds                                         labels p0o2_, p0os_
#   -O3 with a benign source edit (perturb.py), in job A's cache:                         labels p0reorder_, p0field_
#     reorder: two members of the sender channel state swapped
#     field:   one member added to the sender channel state, set at init, never read
# Every build's new router ELFs get the stack check after its first suite and are saved with their generated headers
# under fabric_exp/consolidation/builds/<build>/<config hash>/.
# Flags 13129088, fast dispatch, telemetry off. Tree restored to s79mc on exit, checked with cmp.
cd /home/aagarwal/tt-metal
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
R=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp
H=tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp
P=tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp
V=fabric_exp/variants
OUT=fabric_exp/consolidation/builds
OD=runtime/sfpi/compiler/bin/riscv-tt-elf-objdump
NM=runtime/sfpi/compiler/bin/riscv-tt-elf-nm
swap() { cp $V/$1/fabric_erisc_router_v2.cpp $R && cp $V/$1/fabric_erisc_datamover_channels.hpp $H \
    && cp $V/$1/fabric_erisc_router_speedy_path.hpp $P && echo "== router: $1"; }
restore() { swap s79mc > /dev/null
    cmp -s $V/s79mc/fabric_erisc_router_v2.cpp $R && cmp -s $V/s79mc/fabric_erisc_datamover_channels.hpp $H \
    && cmp -s $V/s79mc/fabric_erisc_router_speedy_path.hpp $P && echo "tree restored to s79mc" || echo "!!! RESTORE FAILED"; }
new_elfs() {  # cache marker-file
    find $1/*/kernels/fabric_erisc_router_v2 -newer $2 -name '*.elf' ! -name '*.xip.elf' 2>/dev/null
}
save_builds() {  # build cache marker-file
    local n=0
    rm -rf $OUT/$1
    for d in $(new_elfs $2 $3 | xargs -r -n1 dirname | sort -u); do
        local h=$(basename $(dirname $d)) k=$(basename $d) c=$(dirname $(dirname $d))
        mkdir -p $OUT/$1/$h && cp $d/$k.elf $OUT/$1/$h/ && cp $c/$h/*.h $c/$h/*.hpp $OUT/$1/$h/ && n=$((n+1))
    done
    echo "saved $n ELFs for $1"
}
stack_check() {  # cache marker-file
    local min=99999 n=0
    for e in $(new_elfs $1 $2); do
        local f=$($OD -d $e | grep -oE 'addi\s+sp,sp,-[0-9]+' | grep -oE '[0-9]+$' | sort -rn | head -3 | paste -sd+ | bc)
        local free=$(( 0xFFB02000 - 0x$($NM $e | awk '$3=="__ldm_bss_end"{print $1}') ))
        local m=$((free - f - 256)); n=$((n+1))
        [ $m -lt $min ] && min=$m && echo "   $(basename $e) $(basename $(dirname $(dirname $e))): frames $f, free $free"
    done
    echo "stack check: $n ELFs, smallest margin $min B"
    [ $n -gt 0 ] || { echo "!!! no router ELFs were built"; echo STOPPED; exit 1; }
    [ $min -ge 0 ] || { echo STOPPED; exit 1; }
}
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null; restore' EXIT
run() {  # label yaml cache opt
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 1200 fabric_exp/run_exp.sh "$1" "$2" \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 \
        TT_METAL_CACHE=$3 TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 TT_METAL_FABRIC_OPT_LEVEL=$4 \
        | grep --line-buffered -v 'Running Test'
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT}"
    [ -n "$res" ] && echo "$res" | grep -q ", 0 failed"
}
suite_yaml() {
    case $1 in
        stream) echo $SAN/test_fabric_stream_sanity.yaml ;; mesh) echo $SAN/test_fabric_mesh_sanity_common.yaml ;;
        val1d) echo $SAN/test_fabric_sanity_at_least_2x2_mesh.yaml ;;
        4352) echo fabric_exp/exp_2d_4352.yaml ;; mix) echo fabric_exp/streams/p3_mixes.yaml ;;
        flushmix) echo fabric_exp/streams/flush_mix.yaml ;; flushmix4) echo fabric_exp/streams/flush_mix4.yaml ;;
        1d) echo fabric_exp/exp_1d_4352.yaml ;;
    esac
}
M=$(mktemp)
build() {  # build-name variant cache opt label-prefix suite...
    local b=$1 v=$2 c=$3 o=$4 l=$5 first=1; shift 5
    mkdir -p $c
    touch $M; sleep 1
    swap $v; sleep 1
    echo "== build $b: router $v, -$o, cache $c"
    for s in "$@"; do
        run ${l}_13129088_$s $(suite_yaml $s) $c $o || { echo STOPPED; exit 1; }
        [ $first = 1 ] && stack_check $c $M && first=0
    done
    save_builds $b $c $M
}
S="stream mesh val1d 4352 mix flushmix flushmix4 1d"
B="4352 mix flushmix flushmix4 1d"
C3=/home/aagarwal/.tt-metal-cache-p0
case "${1:-a}" in
a)  build O2 s79mc /home/aagarwal/.tt-metal-cache-p0-O2 O2 p0o2 $S
    build Os s79mc /home/aagarwal/.tt-metal-cache-p0-Os Os p0os $S
    build reorder s79mc_reorder $C3 O3 p0reorder $B
    build field s79mc_field $C3 O3 p0field $B ;;
esac
rm -f $M
echo ALL_DONE
