#!/bin/bash
# RING_JAM_FIX.md §8 step 1 (ring-jam telemetry), P150: v4txa11j swapped in once; arms off (its counters compiled out:
# v4txa11's images) and on (TT_FABRIC_PHASE_PROFILE=1: the host's phase-profile block; per router and test the counts go
# to fabric_exp/results/<label>_jam.csv through TT_FABRIC_PHASE_PROFILE_CSV).
#   frg_jam_p150_runs.sh val     data-checked validation on both arms: v4/val_ring_p150.yaml (labels frgjv_<arm>_ring)
#   frg_jam_p150_runs.sh PASS    frgap/p150_it_ring.yaml for ARMS in order (default "off on"; "on off" for the
#                                reversed pass), labels frgj<PASS>_<arm>_ring
# One kernel cache per arm. A hang, a failure or a run without a result stops the job (the lb2 job then resets and
# checks the links). No flow-control full-ring row (LoudBox rule, 2026-11-03).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS \
    TT_FABRIC_PHASE_PROFILE TT_FABRIC_PHASE_PROFILE_CSV
P=$1
[ -n "$P" ] || { echo "usage: [ARMS=...] frg_jam_p150_runs.sh val|PASS"; exit 1; }
for f in fabric_exp/v4/val_ring_p150.yaml $CON/frgap/p150_it_ring.yaml; do [ -f $f ] || { echo "no $f"; exit 1; }; done
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
python3 $CON/swap.py apply fabric_exp/variants/v4txa11j $BK || stop
one() {  # label yaml arm
    local label=$1 y=$2 a=$3 rc
    local -a e=(TT_FABRIC_EXP_LOG_SLOTS=1)
    case $a in
        off) ;;
        on) e+=(TT_FABRIC_PHASE_PROFILE=1 TT_FABRIC_PHASE_PROFILE_CSV=$PWD/$out/${label}_jam.csv) ;;
        *) echo "!!! arm $a: off | on"; stop ;;
    esac
    run_cap $label $y /home/aagarwal/.tt-metal-cache-frgj-$a 600 "${e[@]}"; rc=$?
    [ $rc = 2 ] && { echo "!!! $label: hang (captured): stopping"; stop; }
    [ $rc = 0 ] || { echo "!!! $label: rc $rc (failure or no result): stopping"; stop; }
    [ $a = on ] && echo "   $label: $(($(wc -l < $out/${label}_jam.csv 2>/dev/null || echo 1) - 1)) counter rows"
}
if [ "$P" = val ]; then
    for a in off on; do one frgjv_${a}_ring fabric_exp/v4/val_ring_p150.yaml $a; done
else
    for a in ${ARMS:-off on}; do one frgj${P}_${a}_ring $CON/frgap/p150_it_ring.yaml $a; done
fi
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
echo ALL_DONE
