#!/bin/bash
# Phase 1, round 11.
#   inj: which ERISC makes v3l3be1c slower at -Os? (as p1r5_runs.sh c, RESULTS §88). v3l3be1c swapped in, its -O3
#        kernels built in a fresh cache (4352, mix, 1d at -O3), then two mixed sets from offline_build.py injected over
#        them: E0 at -Os with E1 at -O3, and E0 at -O3 with E1 at -Os. Checked: no router recompiles, and the cache
#        still holds the injected ELFs after the runs.                    labels p11inj_, p11e0os_, p11e1os_
#   ka0: the production keepalive: v3a and v3l3be1c at -O3 with TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=0 (the router leaves
#        the firmware's 8000-cycle sequence-update timeout; every other study number uses 1000), benchmarks only.
#                                                                         labels p11ka0a_, p11ka0m_
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
case "$1" in
inj) CI=/home/aagarwal/.tt-metal-cache-p1-inj-v3l3be1c
    rm -rf "${CI:?}"; mkdir -p $CI
    python3 $CON/swap.py apply $V/v3l3be1c $BK || { echo STOPPED; exit 1; }
    for s in 4352 mix 1d; do run p11inj_13129088_$s $(suite_yaml $s) $CI O3 || { echo STOPPED; exit 1; }; done
    for set in v3l3be1c_e0Os_e1O3 v3l3be1c_e0O3_e1Os; do
        l=$([ $set = v3l3be1c_e0Os_e1O3 ] && echo p11e0os || echo p11e1os)
        inject $CI $set || { echo STOPPED; exit 1; }
        for s in 4352 mix 1d; do run ${l}_13129088_$s $(suite_yaml $s) $CI O3 || { echo STOPPED; exit 1; }; done
        no_recompile $l 4352 mix 1d && verify $CI $set || { echo STOPPED; exit 1; }
    done
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; } ;;
ka0) run() {  # jobs_lib.sh's run with the keepalive at the firmware value
        env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 1200 fabric_exp/run_exp.sh "$1" "$2" \
            TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=0 TT_FABRIC_EXP_FLAGS=13129088 \
            TT_METAL_CACHE=$3 TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 TT_METAL_FABRIC_OPT_LEVEL=$4 \
            | grep --line-buffered -v 'Running Test'
        local res=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/$1.log 2>/dev/null)
        echo "$1: ${res:-NO RESULT}"
        [ -n "$res" ] && echo "$res" | grep -q ", 0 failed"
    }
    build v3a_ka0 $V/v3a $C3 O3 p11ka0a $BENCH
    build v3l3be1c_ka0 $V/v3l3be1c $C3 O3 p11ka0m $BENCH ;;
*)  echo "usage: p1r11_runs.sh inj|ka0"; exit 1 ;;
esac
rm -f $M
echo ALL_DONE
