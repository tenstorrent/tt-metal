#!/bin/bash
# Step 1c of the sender cleanup (V4_SENDER_SHAPE.md; 2026-10-29): variants/v4sh3 = v4sh2b with the sender state grouped
# by channel role (the worker channels' state, their interfaces apart, the forwarding table with forwarding channels
# only, one far ring per VC). Its placement change is on VC2 routers (Blackhole 2D with TT_METAL_ENABLE_FABRIC_VC2=1):
# in v4sh2b the table holds VC2's worker, whose interface escapes, so the table and VC0's forwarding records are a frame
# object there. variants/v4sh3m = v4sh3 with VC2's worker state and far ring in one record with its interface (in
# memory: the fallback). VC2 has no benchmark rows of its own, so this runs the census suites that build VC2 routers
# with VC2 on (their traffic is VC0's), v4sh2b, v4sh3 and v4sh3m back to back per suite in one job:
#   two ERISCs: exp_2d_4352 (4352), p3_mixes (mix), exp_burst (burst), v4/exp_torus_y (torusy);
#   one ERISC: exp_2d_4352 (4352_1e).
# Every yaml passed the mock cluster with VC2 on first. sh3_vc2_runs.sh PASS; labels cen<PASS>_<router>_<suite>
# (census_report.py --passes=... --routers=v4sh2b,v4sh3,v4sh3m --ref=v4sh2b). A hang stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
P=$1
[ -n "$P" ] || { echo "usage: sh3_vc2_runs.sh PASS"; exit 1; }
VC2="TT_METAL_ENABLE_FABRIC_VC2=1"
E="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
cache() { echo /home/aagarwal/.tt-metal-cache-sh3vc2-$1; }
RS=${ROUTERS:-v4sh2b v4sh3 v4sh3m}  # ROUTERS overrides (e.g. "v4sh2b v4sh3 v4pl1")
for r in $RS; do c=$(cache $r); rm -rf "${c:?}" "${c:?}-one"; mkdir -p $c $c-one; done
# suite SUITE YAML TIMEOUT CACHE_SUFFIX [ENV...]: the suite on each router in turn
suite() {
    local s=$1 y=$2 t=$3 x=$4 r; shift 4
    for r in $RS; do
        python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop
        run_cap cen${P}_${r}_$s $y $(cache $r)$x $t $VC2 "$@"
        [ $? = 2 ] && stop
        python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
    done
}
echo "##### two ERISCs (job at ${SECONDS} s)"
suite 4352 $(suite_yaml 4352) 300 ""
suite mix $(suite_yaml mix) 300 ""
suite burst $(suite_yaml burst) 300 ""
suite torusy fabric_exp/v4/exp_torus_y.yaml 200 ""
echo "##### one ERISC (job at ${SECONDS} s)"
suite 4352_1e $(suite_yaml 4352) 300 -one $E
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
