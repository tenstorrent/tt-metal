#!/bin/bash
# Report multi-VC throughput (user, 2026-11-02): VC0 and VC2 sharing one link (fabric_exp/pcs3 rows, P150x8, one E/W
# link from chip 0 to chip 1), main's router as shipped against the new router, in the same job.
#   multivc_runs.sh PASS     ROUTERS in order (default "main_fw v4rxg3"; reversed for pass b), SLOTS in order
#                            (default "default v8").
#   Rows at 256 / 1024 / 4352 B:
#     vc02ew   VC0 + VC2 senders, 4M packets each; VC0 finishes first: VC0's rate while VC2 also sends
#     vc02bew  VC0 40M, VC2 2M packets; VC2 finishes first: VC2's rate while VC0 also sends
#     vc2ew    VC2 alone
#   plus ew4352 with VC2 compiled in (VC0 alone). Per-sender rates come from the profiler's log lines (txp_report.py's
#   parse); the summary CSV gives the link (the longest sender's cycles).
#   SLOTS: default = each router's own VC2 allocation (as shipped); v8 = TT_FABRIC_EXP_SLOTS_VC2=17,10,16,8,10, the
#   sender study's equal-ish VC0 / VC2 split, the same for both routers. TT_FABRIC_EXP_LOG_SLOTS=1 logs what each used.
# Labels mvc<PASS>_<router>_<slots>_<row>. A hang or failure stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS \
    TT_FABRIC_EXP_SLOTS_VC2
P=$1
[ -n "$P" ] || { echo "usage: [ROUTERS=...] [SLOTS=...] multivc_runs.sh PASS"; exit 1; }
V1="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
VC2="TT_METAL_ENABLE_FABRIC_VC2=1 TT_FABRIC_EXP_LOG_SLOTS=1"
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
ROWS="ew4352"
for s in 256 1024 4352; do ROWS="$ROWS vc02ew$s vc02bew$s vc2ew$s"; done
for r in ${ROUTERS:-main_fw v4rxg3}; do
    case $r in main_fw) ;; *) python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop ;; esac
    for sl in ${SLOTS:-default v8}; do
        c=/home/aagarwal/.tt-metal-cache-mvc-$r-$sl
        mkdir -p $c
        S=""; [ $sl = v8 ] && S="TT_FABRIC_EXP_SLOTS_VC2=17,10,16,8,10"
        for row in $ROWS; do
            echo "##### $row on $r, slots $sl (job at ${SECONDS} s)"
            case $r in
                main_fw) KA_ENV= run_cap mvc${P}_${r}_${sl}_$row fabric_exp/pcs3/$row.yaml $c 300 $V1 $VC2 $S; rc=$? ;;
                *) run_cap mvc${P}_${r}_${sl}_$row fabric_exp/pcs3/$row.yaml $c 300 $VC2 $S; rc=$? ;;
            esac
            [ $rc = 0 ] || { echo "!!! mvc${P}_${r}_${sl}_$row: rc $rc"; stop; }
        done
    done
    case $r in main_fw) ;; *) python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; } ;; esac
done
echo ALL_DONE
