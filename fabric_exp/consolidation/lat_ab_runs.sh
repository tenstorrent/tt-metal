#!/bin/bash
# Same-job latency A/B (user, 2026-11-02): one-packet round trips (exp_latency.yaml, RESULTS §19: 1-hop E/W and N/S,
# 3-hop E/W; write and fused atomic; 16 B and 4096 B) for main's router and the new router on one host, in one job.
#   lat_ab_runs.sh PASS     runs ROUTERS in order (default "main_fw main v4rxg3"); pass b: give ROUTERS reversed.
#     main_fw  main's router as shipped (kernel v1, flags 0, firmware keepalive)
#     main     the same router at the new router's keepalive (1000 cycles): shows whether the keepalive moves latency
#     <variant> a variant directory swapped into the v2 slot, study settings
# Labels lat<PASS>_<router>; per-sample CSVs results/<label>_latency.csv; summary lat_ab_report.py. A hang stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
P=$1
[ -n "$P" ] || { echo "usage: [ROUTERS=...] lat_ab_runs.sh PASS"; exit 1; }
V1="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
Y=$(suite_yaml lat)
for r in ${ROUTERS:-main_fw main v4rxg3}; do
    c=/home/aagarwal/.tt-metal-cache-latab-$r
    rm -rf "${c:?}"; mkdir -p $c
    echo "##### $r (job at ${SECONDS} s)"
    case $r in
        main_fw) KA_ENV= run_cap lat${P}_$r $Y $c 900 $V1; rc=$? ;;
        main) run_cap lat${P}_$r $Y $c 900 $V1; rc=$? ;;
        *) python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop
           run_cap lat${P}_$r $Y $c 900; rc=$?
           python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; } ;;
    esac
    [ $rc = 2 ] && stop
    [ $rc = 0 ] || { echo "!!! lat${P}_$r: test failures (rc $rc)"; stop; }
done
echo ALL_DONE
