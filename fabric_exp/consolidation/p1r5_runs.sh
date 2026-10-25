#!/bin/bash
# Phase 1, round 5.
#   a: v3l2 (v3l's worker path + inline compile-time forwarding visits; E1 = v3a) -O3: validation (RELAXED),
#      validation + flushed-signal suites (STRICT), benchmarks; hook probe       labels p5l2_, p5l2s_, p5l2hook_
#   b: v3l2 at -O2 and -Os (benchmarks); VC1 at the default level (-Os), -O2, -O3
#                                                                               labels p5l2o2_, p5l2os_, p5vc1l2_*
#   c: which ERISC makes -Os slow? v3l swapped in, its -O3 kernels built in a fresh cache (4352 + mix at -O3, an
#      -O3 repeat), then two mixed sets injected over them (same source; ELFs from offline_build.py at the two
#      levels): E0 at -Os with E1 at -O3, and E0 at -O3 with E1 at -Os.          labels p5inj_, p5e0os_, p5e1os_
#      The JIT reuses the injected ELFs (tree and dependency hashes unchanged); checked: no router recompiles, and
#      the cache still holds the injected ELFs after the runs.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
SETS=/home/aagarwal/.tt-metal-p1-inject
inject() {  # cache set
    local n=0 e d
    for e in $SETS/$2/*/*/*.elf; do
        local hp=${e#$SETS/$2/}; hp=${hp%/*}
        for d in $1/*/kernels/fabric_erisc_router_v2/$hp; do
            [ -f $d/$(basename $e) ] || continue
            cp $e $d/ && rm -f $d/$(basename $e).xip.elf && n=$((n+1))
        done
    done
    echo "injected $n ELFs from $2"
    [ $n -gt 0 ]
}
verify() {  # cache set
    local bad=0 n=0 e d
    for e in $SETS/$2/*/*/*.elf; do
        local hp=${e#$SETS/$2/}; hp=${hp%/*}
        for d in $1/*/kernels/fabric_erisc_router_v2/$hp; do
            [ -f $d/$(basename $e) ] || continue
            n=$((n+1)); cmp -s $e $d/$(basename $e) || bad=$((bad+1))
        done
    done
    echo "verify $2: $n ELFs in the cache, $bad replaced by the JIT"
    [ $n -gt 0 ] && [ $bad -eq 0 ]
}
no_recompile() {  # label-prefix suite...
    local l=$1 s nc; shift
    for s in "$@"; do
        nc=$(grep -a 'g++ compile cmd' fabric_exp/results/${l}_13129088_$s.log | grep -c fabric_erisc_router_v2)
        [ $nc -eq 0 ] || { echo "!!! $nc router kernels recompiled in ${l}_$s"; return 1; }
    done
}
case "${1:-a}" in
a)  build v3l2 $V/v3l2 $C3 O3 p5l2 $VAL $BENCH
    build v3l2_strict $V/v3l2_strict $C3 O3 p5l2s $VAL flushmix flushmix4
    build v3l2_hook $V/v3l2_hook $C3 O3 p5l2hook $BENCH ;;
b)  build v3l2_O2 $V/v3l2 $C2 O2 p5l2o2 $BENCH
    build v3l2_Os $V/v3l2 $CS Os p5l2os $BENCH
    source $CON/vc1_lib.sh
    vc1_variant v3l2 $V/v3l2 p5vc1l2 auto O2 O3 ;;
c)  CI=/home/aagarwal/.tt-metal-cache-p1-inj-v3l
    rm -rf "${CI:?}"; mkdir -p $CI
    python3 $CON/swap.py apply $V/v3l $BK || { echo STOPPED; exit 1; }
    for s in 4352 mix; do run p5inj_13129088_$s $(suite_yaml $s) $CI O3 || { echo STOPPED; exit 1; }; done
    for set in v3l_e0Os_e1O3 v3l_e0O3_e1Os; do
        l=$([ $set = v3l_e0Os_e1O3 ] && echo p5e0os || echo p5e1os)
        inject $CI $set || { echo STOPPED; exit 1; }
        for s in 4352 mix; do run ${l}_13129088_$s $(suite_yaml $s) $CI O3 || { echo STOPPED; exit 1; }; done
        no_recompile $l 4352 mix && verify $CI $set || { echo STOPPED; exit 1; }
    done
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; } ;;
esac
rm -f $M
echo ALL_DONE
