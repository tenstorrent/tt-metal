#!/bin/bash
# FULLRING_MCAST_GAP.md A/B 3 (P150, 400G): ring multicast rows at 3072-4352 B (frgap/p150_ring_mcast_sizes.yaml),
# main's router as shipped against v4rxg3, in the same job. report_size_runs.sh with this yaml.
#   frg_p150_runs.sh PASS     ROUTERS in order (default "main_fw v4rxg3"; reversed for pass b)
#     main_fw   main's router as shipped (kernel v1, flags 0, firmware keepalive)
#     <variant> a variant directory swapped into the v2 slot, study settings (keepalive 1000)
#     <variant>:on  the same with TT_FABRIC_EXP_FLAGS=13129089 (bit 0: v4rxg3sl's local-write skip)
# Labels frg<PASS>_<router>_p150ring (.log, .csv). A hang or failure stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
P=$1
[ -n "$P" ] || { echo "usage: [ROUTERS=...] frg_p150_runs.sh PASS"; exit 1; }
y=$CON/frgap/p150_ring_mcast_sizes.yaml
[ -f $y ] || { echo "no $y"; exit 1; }
V1="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
for r in ${ROUTERS:-main_fw v4rxg3}; do
    c=/home/aagarwal/.tt-metal-cache-frg-${r%:on}$([[ $r == *:on ]] && echo on)
    mkdir -p $c
    echo "##### p150ring on $r (job at ${SECONDS} s)"
    case $r in
        main_fw) KA_ENV= run_cap frg${P}_${r}_p150ring $y $c 900 $V1 TT_FABRIC_EXP_LOG_SLOTS=1; rc=$? ;;
        *:on) v=${r%:on}  # the variant with TT_FABRIC_EXP_FLAGS bit 0 set (v4rxg3sl: skip the local write)
           python3 $CON/swap.py apply fabric_exp/variants/$v $BK || stop
           run_cap frg${P}_${v}on_p150ring $y $c 900 TT_FABRIC_EXP_LOG_SLOTS=1 TT_FABRIC_EXP_FLAGS=13129089; rc=$?
           python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; } ;;
        *) python3 $CON/swap.py apply fabric_exp/variants/$r $BK || stop
           run_cap frg${P}_${r}_p150ring $y $c 900 TT_FABRIC_EXP_LOG_SLOTS=1; rc=$?
           python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; } ;;
    esac
    [ $rc = 0 ] || { echo "!!! frg${P}_${r}_p150ring: rc $rc"; stop; }
done
echo ALL_DONE
