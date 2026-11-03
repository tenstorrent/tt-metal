#!/bin/bash
# Report packet-size sweep at 400G (user, 2026-11-02): the census's two-ERISC suites (2D, 1D, ring, Y-torus) at
# 256 / 1024 / 2048 / 3072 / 4352 B (fabric_exp/report/sweep/<suite>_sizes.yaml, generated from the census yamls),
# main's router as shipped against the new router, in the same job.
#   report_size_runs.sh PASS SUITE...   SUITE in 2d 1d ring torusy; ROUTERS in order (default "main_fw v4rxg3"; give
#                                       them reversed for pass b).
#     main_fw   main's router as shipped (kernel v1, flags 0, firmware keepalive)
#     <variant> a variant directory swapped into the v2 slot, study settings (keepalive 1000)
# Labels rsz<PASS>_<router>_<suite> (.log, .csv). A hang or failure stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
P=$1; shift
[ -n "$P" ] && [ $# -gt 0 ] || { echo "usage: [ROUTERS=...] report_size_runs.sh PASS SUITE..."; exit 1; }
V1="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
for s in "$@"; do
    y=fabric_exp/report/sweep/${s}_sizes.yaml
    [ -f $y ] || { echo "no $y"; exit 1; }
    for r in ${ROUTERS:-main_fw v4rxg3}; do
        c=/home/aagarwal/.tt-metal-cache-rsz-$r
        mkdir -p $c
        echo "##### $s on $r (job at ${SECONDS} s)"
        case $r in
            main_fw) KA_ENV= run_cap rsz${P}_${r}_$s $y $c 900 $V1; rc=$? ;;
            *) python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop
               run_cap rsz${P}_${r}_$s $y $c 900; rc=$?
               python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; } ;;
        esac
        [ $rc = 0 ] || { echo "!!! rsz${P}_${r}_$s: rc $rc"; stop; }
    done
done
echo ALL_DONE
