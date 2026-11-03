#!/bin/bash
# Route-preload study probe (V4_ROUTE_PRELOAD.md §8, 2026-11-02; drafts, not for adoption).
#   rp_probe_runs.sh val VARIANT          mesh sanity (two ERISCs), then rxo0_inline_runs.sh's inline-write and flushed
#                                         atomic gtests (1D, 2D) on VARIANT: the correctness guard for the draft's
#                                         word-view cast. Labels prlv_<variant>_mesh, rxoi_<router>_<1d|2d>.
#   rp_probe_runs.sh lat PASS ROUTER...   exp_latency.yaml (RESULTS §19) on each router in turn, a fresh cache each.
#                                         Labels prllat<PASS>_<router> (.log, _latency.csv).
#   rp_probe_runs.sh codec PASS ROUTER... the census's 2D 4352 and mix suites (the codec rows: E/W and forwarded
#                                         atomic / fused / scatter, multicast scatter), each router in turn. Labels
#                                         cen<PASS>_<router>_<4352|mix>, read by census_report.py --passes.
# A failure or hang stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
MODE=$1; shift
case $MODE in
val)
    V=$1
    [ -n "$V" ] && [ -d fabric_exp/variants/$V ] || { echo "usage: rp_probe_runs.sh val VARIANT"; exit 1; }
    C=/home/aagarwal/.tt-metal-cache-prlv-$V
    rm -rf "${C:?}"; mkdir -p $C
    python3 $CON/swap.py apply fabric_exp/variants/$V $BK || stop
    run_cap prlv_${V}_mesh $(suite_yaml mesh) $C 300; rc=$?
    [ $rc = 0 ] || { echo "!!! prlv_${V}_mesh failed (rc $rc)"; stop; }
    grep -aq "Tests: 96 executed, 0 failed" $out/prlv_${V}_mesh.log || { echo "!!! prlv_${V}_mesh: not 96/96"; stop; }
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
    bash $CON/rxo0_inline_runs.sh $V || { echo STOPPED; exit 1; }
    ;;
lat)
    P=$1; shift
    [ -n "$P" ] && [ $# -gt 0 ] || { echo "usage: rp_probe_runs.sh lat PASS ROUTER..."; exit 1; }
    for r in "$@"; do
        C=/home/aagarwal/.tt-metal-cache-prllat-$r
        rm -rf "${C:?}"; mkdir -p $C
        python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop
        run_cap prllat${P}_$r $(suite_yaml lat) $C 900; rc=$?
        [ $rc = 0 ] || { echo "!!! prllat${P}_$r failed (rc $rc)"; stop; }
        python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
    done
    ;;
codec)
    P=$1; shift
    [ -n "$P" ] && [ $# -gt 0 ] || { echo "usage: rp_probe_runs.sh codec PASS ROUTER..."; exit 1; }
    for r in "$@"; do
        C=/home/aagarwal/.tt-metal-cache-prlcodec-$r
        rm -rf "${C:?}"; mkdir -p $C
        python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop
        for s in 4352 mix; do
            run_cap cen${P}_${r}_$s $(suite_yaml $s) $C 300; rc=$?
            [ $rc = 0 ] || { echo "!!! cen${P}_${r}_$s failed (rc $rc)"; stop; }
        done
        python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
    done
    ;;
*) echo "usage: rp_probe_runs.sh val VARIANT | lat PASS ROUTER... | codec PASS ROUTER..."; exit 1 ;;
esac
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
