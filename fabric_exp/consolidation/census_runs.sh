#!/bin/bash
# Perf census after the features (2026-10-29; user: tackle the perf gaps, first see where every config stands). One job
# per GROUP and PASS (census_runs.sh GROUP PASS), each suite run by every router back to back in the same job:
#   main_fw   main's router as shipped: kernel v1, flags 0 (none of the study's experiments in it), firmware keepalive
#   main      the same router at the study's keepalive (1000 cycles), so router and keepalive separate
#   v3qm      the v3 lead, variants/v3qm = v3q with the post-merge packet header path (nothing else; builds all 336
#             post-merge production configs offline), study settings
#   v4trm     the v4 lead (every production config byte-identical to v4rs, where perf was parked, but the 22 VC1 senders),
#             study settings
# GROUP two1, two2 (two ERISCs): two1 exp_2d_4352 (4352), exp_1d_4352 (1d: lines, the ring row, NeighborExchange = the
#   speedy path), p3_mixes (mix); pass a of two1 starts with v3qm's data-checked validation (mesh sanity,
#   val_ring_p150), hang capture armed. two2 exp_burst (burst), v4/exp_ring (ring), v4/exp_torus_y (torusy).
# GROUP one: one ERISC (TT_METAL_DISABLE_FABRIC_TWO_ERISC=1) exp_2d_4352 (4352_1e), exp_1d_4352 (1d_1e), v4/exp_ring_se
#   (rse); mux mode v4/exp_mux (mux); channel-trimmed exp_2d_trim_4352 (trim; main and v4trm only).
# Labels cen<pass>_<router>_<suite>; compare with same_job_compare.py. A hang stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
G=$1 P=$2
case $G in two1|two2|one) ;; *) echo "usage: census_runs.sh two1|two2|one PASS"; exit 1 ;; esac
[ -n "$P" ] || { echo "usage: census_runs.sh two1|two2|one PASS"; exit 1; }
E="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
V1="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
cache() { echo /home/aagarwal/.tt-metal-cache-cen-$1-$G; }
for r in main_fw main v3qm v4trm; do c=$(cache $r); rm -rf "${c:?}"; mkdir -p $c; done

# one ROUTER SUITE YAML TIMEOUT [ENV...]: one suite on one router (the v2-slot routers must be swapped in by the caller)
one() {
    local r=$1 s=$2 y=$3 t=$4; shift 4
    case $r in
        main_fw) KA_ENV= run_cap cen${P}_${r}_$s $y $(cache $r) $t $V1 "$@" ;;
        main) run_cap cen${P}_${r}_$s $y $(cache $r) $t $V1 "$@" ;;
        *) run_cap cen${P}_${r}_$s $y $(cache $r) $t "$@" ;;
    esac
    [ $? = 2 ] && stop
    return 0
}
# suite SUITE YAML TIMEOUT ROUTERS [-- ENV...]: the suite on each router in turn
suite() {
    local s=$1 y=$2 t=$3 routers=$4 r; shift 4
    for r in $routers; do
        case $r in
            v3qm|v4trm) python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop
                        one $r $s $y $t "$@"
                        python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; } ;;
            *) one $r $s $y $t "$@" ;;
        esac
    done
}
ALL="main_fw main v3qm v4trm"

if [ $G = two1 ]; then
    if [ $P = a ]; then
        echo "##### v3qm validation (job at ${SECONDS} s)"
        python3 $CON/swap.py apply fabric_exp/variants/v3qm $BK || stop
        run_cap cenval_v3qm_mesh $(suite_yaml mesh) $(cache v3qm) 300; [ $? = 2 ] && stop
        run_cap cenval_v3qm_ring fabric_exp/v4/val_ring_p150.yaml $(cache v3qm) 120; [ $? = 2 ] && stop
        python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
    fi
    for s in 4352 1d mix; do
        echo "##### $s (job at ${SECONDS} s)"
        suite $s $(suite_yaml $s) 300 "$ALL"
    done
elif [ $G = two2 ]; then
    echo "##### burst, ring, torusy (job at ${SECONDS} s)"
    suite burst $(suite_yaml burst) 300 "$ALL"
    suite ring fabric_exp/v4/exp_ring.yaml 200 "$ALL"
    suite torusy fabric_exp/v4/exp_torus_y.yaml 200 "$ALL"
else
    echo "##### one ERISC (job at ${SECONDS} s)"
    suite 4352_1e $(suite_yaml 4352) 300 "$ALL" $E
    suite 1d_1e $(suite_yaml 1d) 300 "$ALL" $E
    suite rse fabric_exp/v4/exp_ring_se.yaml 300 "$ALL" $E
    echo "##### mux (job at ${SECONDS} s)"
    suite mux fabric_exp/v4/exp_mux.yaml 400 "$ALL"
    echo "##### trimmed (job at ${SECONDS} s)"
    suite trim fabric_exp/exp_2d_trim_4352.yaml 300 "main_fw main v4trm"
fi
rm -f $M
echo "ALL_DONE (job at ${SECONDS} s)"
