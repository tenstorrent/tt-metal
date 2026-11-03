#!/bin/bash
# VC0 + VC2 combined rate measured in one run (user, 2026-11-02): pcs3/vc02m/<router>_<slots>_<size>.yaml from
# consolidation/mk_mvc_matched.py, the VC2 count set so both senders finish together. The summary CSV is then the
# link's combined rate with both VCs active for the whole run; the per-sender lines give each VC's share and the
# finish gap. Checks the two-run sum in multivc_runs.sh (VC0's rate from vc02ew + VC2's from vc02bew).
#   multivc_matched_runs.sh PASS    ROUTERS in order (default "main_fw v4rxg3"; reversed for pass b), SLOTS in order
#                                   (default "default v8").
# Labels mvm<PASS>_<router>_<slots>_<size>. A hang or failure stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS \
    TT_FABRIC_EXP_SLOTS_VC2
P=$1
[ -n "$P" ] || { echo "usage: [ROUTERS=...] [SLOTS=...] multivc_matched_runs.sh PASS"; exit 1; }
V1="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
VC2="TT_METAL_ENABLE_FABRIC_VC2=1 TT_FABRIC_EXP_LOG_SLOTS=1"
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
for r in ${ROUTERS:-main_fw v4rxg3}; do
    case $r in main_fw) ;; *) python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop ;; esac
    for sl in ${SLOTS:-default v8}; do
        c=/home/aagarwal/.tt-metal-cache-mvc-$r-$sl
        mkdir -p $c
        S=""; [ $sl = v8 ] && S="TT_FABRIC_EXP_SLOTS_VC2=17,10,16,8,10"
        for s in 256 1024 4352; do
            y=fabric_exp/pcs3/vc02m/${r}_${sl}_$s.yaml
            [ -f $y ] || { echo "no $y"; stop; }
            echo "##### $y (job at ${SECONDS} s)"
            case $r in
                main_fw) KA_ENV= run_cap mvm${P}_${r}_${sl}_$s $y $c 300 $V1 $VC2 $S; rc=$? ;;
                *) run_cap mvm${P}_${r}_${sl}_$s $y $c 300 $VC2 $S; rc=$? ;;
            esac
            [ $rc = 0 ] || { echo "!!! mvm${P}_${r}_${sl}_$s: rc $rc"; stop; }
        done
    done
    case $r in main_fw) ;; *) python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; } ;; esac
done
echo ALL_DONE
