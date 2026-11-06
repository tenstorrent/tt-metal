#!/bin/bash
# Report data refresh at 400G, the rows that need the study's test harness (one lb2 job; never a09u32; user 2026-11-05):
# multi-VC throughput and the stream mixes, main as shipped against the port (PR tip) on one P150x8 LoudBox, same job,
# two passes in reversed router order, telemetry off for both.
#   main  /data/aagarwal/lb2/trees/rf_mainh_d2bd: d2bd12fcf1f      both + the A/B harness only (local, never committed,
#   port  /data/aagarwal/lb2/trees/rf_porth_d51e: d51e42cd78a      same on both trees; router images untouched):
#         the TT_FABRIC_NO_BW_TELEMETRY switch, streams/port/test_infra_6dad.patch (the study's test infra: streams,
#         per-sender profiler lines, inter-mesh profiling), adapter_vc2_credits_6dad.patch (VC2 benchmark senders) and
#         the diagnostic launch delay (streams/port/diag_launch_delay.patch, context rebased: refresh/diag_launch_delay_rf.patch)
# Rows (as the multi-VC jobs 554/555 + 561 and 571/572 ran them on the study tree):
#   vc2   VC0 + VC2 on one 2D E/W link (chip 0 -> 1; TT_METAL_ENABLE_FABRIC_VC2=1): pcs3/ew4352 (VC0 alone, VC2 compiled
#         in), vc02ew<S> (VC0's rate under contention), vc02bew<S> (VC2's), vc2ew<S> (VC2 alone) at 256 / 1024 / 4352 B;
#         then vc02m<S>: counts matched from this job's pass-a rates (rf_matched.py), the link's combined rate in one run
#   mvc3  the VC1 split (two 2x2 meshes, two ranks through tt-run; refresh/mvc/rb_p150_*.yaml), vc1/mvc3 rows: sets
#         B_main (main), B_new (port) VC0 + VC1; A_d (port, VC2 on, TT_METAL_FABRIC_OPT_LEVEL=O3: fits, mock-vetted;
#         the port's own level for VC1 + VC2 is -Os) VC0 + VC1 + VC2; groups base, lag1, lag0 (200 ms launch delay);
#         then match (rf_matched.py from pass a)
#   mix   streams/p3_mixes.yaml (the 4352 B mixes, pure writes / scatter / fused / atomics: EW, NS, FWD)
# Pass a: data-checked validation per router (pval2d_4352), vc2 main, port; mvc3 B_main, B_new, A_d; mix main, port;
# matched counts generated; vc2 matched main, port; mvc3 match B_main, B_new, A_d. Pass b: the same reversed.
# Labels rfh<a|b>_<main|port>_<row> (vc2 rows: ew4352vc2, vc02ew<S>, vc02bew<S>, vc2ew<S>, vc02m<S>; mix), rfh<a|b>_<SET>_<group> (mvc3). The router builds' generated headers are
# kept (rfh_<router>_routers.tgz: slot allocations). A hang stops the job; a failed run is reported and the job goes on.
#   rf_lb_harness.sh        DRY=1: print the runs only
cd /home/aagarwal/tt-metal || exit 1
out=fabric_exp/results
CON=fabric_exp/consolidation
RFD=fabric_exp/report/refresh
DRY=${DRY:-0}
P=rfh
declare -A TREE=([main]=/home/aagarwal/tt-metal-rf-mainh [port]=/home/aagarwal/tt-metal-rf-porth)
VC2ENV="TT_METAL_ENABLE_FABRIC_VC2=1"
D=TT_FABRIC_TEST_DIAG_LAUNCH_DELAY_MS=200
for f in fabric_exp/pcs3/ew4352.yaml fabric_exp/streams/p3_mixes.yaml fabric_exp/port_slots/yamls/pval2d_4352.yaml \
    $RFD/mvc/rb_p150_std.yaml $RFD/mvc/rb_p150_lag0.yaml $RFD/rf_matched.py; do [ -f $f ] || { echo "!!! no $f"; exit 1; }; done
for s in 256 1024 4352; do for r in vc02ew vc02bew vc2ew; do [ -f fabric_exp/pcs3/$r$s.yaml ] || { echo "!!! no $r$s"; exit 1; }; done; done
for s in B_main B_new A_d; do for g in base lag1 lag0; do [ -f fabric_exp/vc1/mvc3/mvc3_${s}_$g.yaml ] || { echo "!!! no mvc3 $s $g"; exit 1; }; done; done
[ "$DRY" = 1 ] || source $RFD/rf_lib.sh || { echo "!!! rf_lib.sh"; exit 1; }
cache() { echo /home/aagarwal/.tt-metal-cache-rfh-$1-$2; }

if [ "$DRY" != 1 ]; then
    [ "$(hostname)" != bh-lb-120-a09u32 ] || { echo "!!! a09u32 holds the build worktrees: run this on an lb2 node"; exit 1; }
    mkdir -p $out
    rf_unpack main ${TREE[main]} $G/rf_mainh_d2bd.tar.zst d2bd12fcf1f
    rf_unpack port ${TREE[port]} $G/rf_porth_d51e.tar.zst d51e42cd78a
    for pass in a b; do for k in main port B_main B_new A_d; do c=$(cache $k $pass); rm -rf "${c:?}"; mkdir -p $c; done; done
fi
chk() {  # label rc
    [ $2 = 2 ] && { echo "!!! $1 hung (captured): stopping"; exit 1; }
    [ $2 = 0 ] || echo "!!! $1 failed (rc $2): going on"
    return 0
}
one() {  # pass router label-suffix yaml timeout env...
    local pass=$1 r=$2 lab=${P}$1_$2_$3 y=$4 t=$5; shift 5
    if [ "$DRY" = 1 ]; then echo "   $lab: ${TREE[$r]} $y (timeout $t) $*"; return 0; fi
    rf_run $lab ${TREE[$r]} $(cache $r $pass) $y $t "$@"; chk $lab $?
}
vc2() {  # pass router
    one $1 $2 ew4352vc2 fabric_exp/pcs3/ew4352.yaml 300 $VC2ENV
    for s in 256 1024 4352; do for row in vc02ew vc02bew vc2ew; do
        one $1 $2 $row$s fabric_exp/pcs3/$row$s.yaml 300 $VC2ENV
    done; done
}
vc2m() {  # pass router: the matched rows (pass a's counts)
    for s in 256 1024 4352; do one $1 $2 vc02m$s $out/${P}a_$2_vc02m$s.yaml 300 $VC2ENV; done
}
mvc3() {  # pass set group...
    local pass=$1 s=$2 r=port env=() g y rb lab; shift 2
    [ $s = B_main ] && r=main
    [ $s = A_d ] && env=($VC2ENV TT_METAL_FABRIC_OPT_LEVEL=O3)
    for g in "$@"; do
        y=fabric_exp/vc1/mvc3/mvc3_${s}_$g.yaml; rb=$RFD/mvc/rb_p150_std.yaml; local e=("${env[@]}")
        case $g in lag1) e+=($D) ;; lag0) e+=($D); rb=$RFD/mvc/rb_p150_lag0.yaml ;; match) y=$out/${P}a_${s}_match.yaml ;; esac
        lab=${P}${pass}_${s}_$g
        if [ "$DRY" = 1 ]; then echo "   $lab: ${TREE[$r]} $y $(basename $rb) ${e[*]}"; continue; fi
        rf_run2 $lab ${TREE[$r]} $(cache $s $pass) $y $PWD/$rb 600 "${e[@]}"; chk $lab $?
    done
}
mix() { one $1 $2 mix fabric_exp/streams/p3_mixes.yaml 900; }
matched() {
    [ "$DRY" = 1 ] && { echo "   rf_matched.py vc02 / mvc3 from pass a"; return 0; }
    python3 $RFD/rf_matched.py vc02 $out ${P}a main port 2>&1 | sed 's/^/   /' || true
    python3 $RFD/rf_matched.py mvc3 $out ${P}a B_main B_new A_d 2>&1 | sed 's/^/   /' || true
}

echo "##### pass a (job at ${SECONDS} s)"
one a main val2d fabric_exp/port_slots/yamls/pval2d_4352.yaml 300
one a port val2d fabric_exp/port_slots/yamls/pval2d_4352.yaml 300
vc2 a main; vc2 a port
mvc3 a B_main base lag1 lag0; mvc3 a B_new base lag1 lag0; mvc3 a A_d base lag1 lag0
mix a main; mix a port
matched
vc2m a main; vc2m a port
mvc3 a B_main match; mvc3 a B_new match; mvc3 a A_d match
echo "##### pass b (job at ${SECONDS} s)"
mvc3 b A_d match; mvc3 b B_new match; mvc3 b B_main match
vc2m b port; vc2m b main
mix b port; mix b main
mvc3 b A_d lag0 lag1 base; mvc3 b B_new lag0 lag1 base; mvc3 b B_main lag0 lag1 base
vc2 b port; vc2 b main
[ "$DRY" = 1 ] || for k in main port B_main B_new A_d; do rf_keep_cache $(cache $k a) ${P}_$k; done
echo "ALL_DONE (job at ${SECONDS} s)"
