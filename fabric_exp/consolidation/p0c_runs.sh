#!/bin/bash
# Phase 0, job C, on bh-lb-120-a08u24 (broker job: no resets, output streamed, heartbeat). s79m's router built with a
# changed compiler setting that stands in for a toolchain update, on hardware, by ELF injection:
#   1. 4352 / mix / flushmix / flushmix4 / 1D at -O3 in a fresh kernel cache $CI (an -O3 repeat)    label p0inj_
#   2. for each setting, copy offline_build.py's ELFs (built from the same s79mc source with one extra compiler flag,
#      $SRC/<setting>/<hash>/<proc>/<proc>.elf) over the router ELFs in $CI and rerun the suites. The tree and the
#      cache's dependency hashes are unchanged, so the JIT reuses the injected ELFs; the script checks that no router
#      kernel was recompiled and that the cache still holds the injected ELFs after each run.   labels p0f_<setting>_
# Settings: ira-algorithm=priority, no-schedule-insns2, no-gcse, no-tree-loop-im (offline: each changes E0's and E1's
# code in every config, codegen_diff.py).
# Job 208 ran this with a shell-variable clash that made `verify` check nothing (fixed here); the injection was checked
# by hand afterwards: no router recompiles, the cache held the last injected set, and the runtime's .xip.elf load
# dumps carried the injected code.
# Flags 13129088, fast dispatch, telemetry off. The tree is not modified; checked against s79mc on exit.
cd /home/aagarwal/tt-metal
R=tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router_v2.cpp
H=tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp
P=tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_speedy_path.hpp
V=fabric_exp/variants
CI=/home/aagarwal/.tt-metal-cache-p0-inj
SRC=/home/aagarwal/.tt-metal-p0-offline
check_tree() { cmp -s $V/s79mc/fabric_erisc_router_v2.cpp $R && cmp -s $V/s79mc/fabric_erisc_datamover_channels.hpp $H \
    && cmp -s $V/s79mc/fabric_erisc_router_speedy_path.hpp $P && echo "tree is s79mc" || { echo "!!! TREE IS NOT s79mc"; false; }; }
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null; check_tree' EXIT
run() {  # label yaml
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 1200 fabric_exp/run_exp.sh "$1" "$2" \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 \
        TT_METAL_CACHE=$CI TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 TT_METAL_FABRIC_OPT_LEVEL=O3 \
        | grep --line-buffered -v 'Running Test'
    local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
    echo "$1: ${res:-NO RESULT}"
    [ -n "$res" ] && echo "$res" | grep -q ", 0 failed"
}
suite_yaml() {
    case $1 in
        4352) echo fabric_exp/exp_2d_4352.yaml ;; mix) echo fabric_exp/streams/p3_mixes.yaml ;;
        flushmix) echo fabric_exp/streams/flush_mix.yaml ;; flushmix4) echo fabric_exp/streams/flush_mix4.yaml ;;
        1d) echo fabric_exp/exp_1d_4352.yaml ;;
    esac
}
suites() {  # label-prefix suite...
    local l=$1 s; shift
    for s in "$@"; do
        run ${l}_13129088_$s $(suite_yaml $s) || { echo STOPPED; exit 1; }
        local nc=$(grep 'g++ compile cmd' fabric_exp/results/${l}_13129088_$s.log | grep -c fabric_erisc_router_v2)
        [ "$l" = p0inj ] || [ $nc -eq 0 ] || { echo "!!! $nc router kernels recompiled in ${l}_$s"; echo STOPPED; exit 1; }
    done
}
inject() {  # setting: copy its ELFs over the cache's; echo how many
    local n=0 m=0
    for e in $SRC/$1/*/*/*.elf; do
        local hp=${e#$SRC/$1/}; hp=${hp%/*}
        for d in $CI/*/kernels/fabric_erisc_router_v2/$hp; do
            [ -f $d/$(basename $e) ] || continue
            cp $e $d/ && rm -f $d/$(basename $e).xip.elf && n=$((n+1))
        done
    done
    echo "injected $n ELFs for $1"
    [ $n -gt 0 ]
}
verify() {  # setting: the cache still holds the injected ELFs
    local bad=0 n=0
    for e in $SRC/$1/*/*/*.elf; do
        local hp=${e#$SRC/$1/}; hp=${hp%/*}
        for d in $CI/*/kernels/fabric_erisc_router_v2/$hp; do
            [ -f $d/$(basename $e) ] || continue
            n=$((n+1)); cmp -s $e $d/$(basename $e) || bad=$((bad+1))
        done
    done
    echo "verify $1: $n ELFs, $bad replaced by the JIT"
    [ $n -gt 0 ] && [ $bad -eq 0 ]
}
B="4352 mix flushmix flushmix4 1d"
check_tree || exit 1
case "${1:-a}" in
a)  rm -rf $CI; mkdir -p $CI
    suites p0inj $B
    for setting in ira-algorithm_priority no-schedule-insns2 no-gcse no-tree-loop-im; do
        inject $setting || { echo STOPPED; exit 1; }
        suites p0f_$setting $B
        verify $setting || { echo STOPPED; exit 1; }
    done ;;
esac
echo ALL_DONE
