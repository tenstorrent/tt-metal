#!/bin/bash
# Flush validation of 7326d59a864 (3efd636e8df rebased) on the cleanup tip de8a2ac17a8, a09u32 through the tt-device
# broker (coordinator 2026-11-05; local, never committed). One data-checked pass, tip only, no flow-controlled full-ring
# row. Tree: /home/aagarwal/tt-metal-gate-de8a (sources at de8a2ac17a8; host build linked from tt-metal-gate-80118, whose
# host sources equal de8a2ac17a8's: git diff 80118e29656 de8a2ac17a8 touches router_receiver.hpp and
# router_single_erisc.hpp only). Suites in the order given; stops at the first run with no result line or a failure, and
# leaves a STOP marker so later broker jobs of this set exit at once.
#   [FLUSHVAL_TREE=... FLUSHVAL_TAG=... FLUSHVAL_SHA=...] flushval_a09.sh SUITE...
#   suites: val (val_ring_p150, first), atomics, rows, torus, atomics_rest, rows_bench (p150_it_ring as proven: benchmark), mesh (stock
#   test_fabric_mesh_sanity_common.yaml), mesh_1e (the same, one ERISC: TT_METAL_DISABLE_FABRIC_TWO_ERISC=1)
cd /home/aagarwal/tt-metal || exit 1
W=${FLUSHVAL_TREE:-/home/aagarwal/tt-metal-gate-de8a}; B=$W/build_Release
TAG=${FLUSHVAL_TAG:-de8a}; SHA=${FLUSHVAL_SHA:-de8a2ac17a8}
OUT=fabric_exp/results; D=fabric_exp/pr_perf/flushda
STOP=$OUT/flushval_$TAG.STOP
SAN=$W/tests/tt_metal/tt_fabric/test_infra/test_yamls
[ -e $STOP ] && { echo "!!! $STOP exists (an earlier run of this set stopped): not running"; exit 1; }
[ "$(git -C $W rev-parse --short=11 HEAD)" = $SHA ] && [ -z "$(git -C $W status --porcelain --untracked-files=no)" ] \
    || { echo "!!! $W is not a clean $SHA"; exit 1; }
(while sleep 60; do echo "heartbeat $(date +%T)"; done) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
run() {  # label timeout yaml [args...]
    local label=flushval_${TAG}_$1 to=$2 y=$3; shift 3
    local log=$OUT/$label.log yp=$y
    [[ $y = /* ]] || yp=/home/aagarwal/tt-metal/$y
    [ -f $yp ] || { echo "!!! no $yp"; touch $STOP; exit 1; }
    echo "=== $label ($(date +%T)): $(basename $y) $* timeout $to s"
    (cd $W && env -u TT_METAL_SLOW_DISPATCH_MODE ${RUN_ENV:-} TT_METAL_HOME=$W TT_METAL_RUNTIME_ROOT=$W \
        TT_METAL_CACHE=/home/aagarwal/.tt-metal-cache-flushval-$TAG TT_FABRIC_NO_BW_TELEMETRY=1 \
        LD_LIBRARY_PATH=$B/tt_metal:$B/tt_stl:$B/lib:$B/tt_metal/third_party/umd/lib \
        timeout $to $B/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config $yp "$@" 2>&1) \
        | tee $log | grep --line-buffered -aE "Running Test:|TT_FATAL|TT_THROW|FAILED|mismatch|Tests: [0-9]+ executed"
    local rc=${PIPESTATUS[0]}
    local res; res=$(sed 's/\x1b\[[0-9;]*m//g' $log | grep -aoE 'Tests: [0-9]+ executed, [0-9]+ failed' | tail -1)
    echo "$label: exit $rc; ${res:-NO RESULT}"
    if [ $rc = 124 ]; then echo "!!! $label timed out (hang?)"; touch $STOP; exit 1; fi
    if [ $rc != 0 ] || ! echo "$res" | grep -q ", 0 failed"; then echo "!!! $label failed: stopping"; touch $STOP; exit 1; fi
}
for s in "$@"; do
    case $s in
        val) run val 900 fabric_exp/v4/val_ring_p150.yaml ;;
        atomics) run atomics 900 $D/flushval_ring_atomics.yaml ;;
        rows) run rows 1200 $D/flushval_ring_rows.yaml ;;
        atomics_rest) run atomics_rest 900 $D/flushval_ring_atomics_rest.yaml ;;
        rows_bench) run rows_bench 1200 fabric_exp/consolidation/frgap/p150_it_ring.yaml ;;
        mesh) run mesh 1200 $SAN/test_fabric_mesh_sanity_common.yaml ;;
        mesh_1e) RUN_ENV=TT_METAL_DISABLE_FABRIC_TWO_ERISC=1 run mesh_1e 1200 $SAN/test_fabric_mesh_sanity_common.yaml ;;
        torus) run torus 1400 $SAN/test_fabric_2d_torus_stability.yaml --filter name.AllToAll2DTorus_short ;;
        *) echo "!!! unknown suite $s"; exit 1 ;;
    esac
done
echo "ALL_DONE $*"
