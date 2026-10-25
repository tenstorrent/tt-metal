#!/bin/bash
# Phase 0, job A, on bh-lb-120-a08u24 (broker job: no resets, output streamed, heartbeat). The tree's router (s79mc:
# s79m with the unused cold-helper templates removed) built three ways with no source change:
#   -O3 (production default without VC1)   validation (stream, mesh, 1D sanity) + 4352 / mix / flushmix / flushmix4 / 1D
#   -O2                                     same                                                   labels p0o3_, p0o2_, p0os_
#   -Os (production default with VC1)       same
# A fresh kernel cache (TT_METAL_CACHE=/home/aagarwal/.tt-metal-cache-p0) forces every kernel to compile, and
# TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 logs each compile and link command into the suite's log, for offline builds.
# Each opt level's router ELFs get the stack check after its first suite and are saved with their generated headers
# under fabric_exp/consolidation/builds/<opt>/<config hash>/.
# Flags 13129088, fast dispatch, telemetry off. The tree is not modified; checked against s79mc on exit.
cd /home/aagarwal/tt-metal
SAN=tests/tt_metal/tt_fabric/test_infra/test_yamls
R=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp
H=tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp
P=tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp
V=fabric_exp/variants
C=/home/aagarwal/.tt-metal-cache-p0
OUT=fabric_exp/consolidation/builds
OD=runtime/sfpi/compiler/bin/riscv-tt-elf-objdump
NM=runtime/sfpi/compiler/bin/riscv-tt-elf-nm
check_tree() { cmp -s $V/s79mc/fabric_erisc_router_v2.cpp $R && cmp -s $V/s79mc/fabric_erisc_datamover_channels.hpp $H \
    && cmp -s $V/s79mc/fabric_erisc_router_speedy_path.hpp $P && echo "tree is s79mc" || echo "!!! TREE CHANGED"; }
new_elfs() {  # marker-file
    find $C/*/kernels/fabric_erisc_router_v2 -newer $1 -name '*.elf' ! -name '*.xip.elf' 2>/dev/null
}
save_builds() {  # opt marker-file
    local n=0
    rm -rf $OUT/$1
    for d in $(new_elfs $2 | xargs -r -n1 dirname | sort -u); do
        local h=$(basename $(dirname $d)) k=$(basename $d) c=$(dirname $(dirname $d))
        mkdir -p $OUT/$1/$h && cp $d/$k.elf $OUT/$1/$h/ && cp $c/$h/*.h $c/$h/*.hpp $OUT/$1/$h/ && n=$((n+1))
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
trap 'kill $HB 2>/dev/null; check_tree' EXIT
run() {  # label yaml opt
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 1200 fabric_exp/run_exp.sh "$1" "$2" \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 \
        TT_METAL_CACHE=$C TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 TT_METAL_FABRIC_OPT_LEVEL=$3 \
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
opt_level() {  # opt label-prefix suite...
    local o=$1 l=$2 first=1; shift 2
    touch $M; sleep 1
    echo "== opt level $o"
    for s in "$@"; do
        run ${l}_13129088_$s $(suite_yaml $s) $o || { echo STOPPED; exit 1; }
        [ $first = 1 ] && stack_check $M && first=0
    done
    save_builds $o $M
}
check_tree
mkdir -p $C
S="stream mesh val1d 4352 mix flushmix flushmix4 1d"
case "${1:-a}" in
a)  opt_level O3 p0o3 $S
    opt_level O2 p0o2 $S
    opt_level Os p0os $S ;;
esac
rm -f $M
echo ALL_DONE
