#!/bin/bash
# FULLRING_MCAST_GAP.md §8.14 (injection throttle), P150: v4rxg3it swapped in once; arms set by
# TT_FABRIC_EXP_DRISC_DELAY = (1 << 16) | (T << 8) | K (unset = off = v4rxg3's images).
#   frg_it_p150_runs.sh val          data-checked validation per arm (ARMS, default "base A B C"), the rows v4rxg3 passes on
#                                    P150: v4/val_ring_p150.yaml (stock ring sanity rows), torus/val_torus_xy_p150.yaml on
#                                    the 2x4 and the 4x2 (fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto) XY torus.
#                                    No flow-control full-ring row on a LoudBox (coordinator, 2026-11-03): frgap/
#                                    p150_frg_val_ring.yaml's FrgValFullRingMcast was killed twice ~20 s into "Waiting for
#                                    programs", the switch off too (FULLRING_MCAST_GAP.md §8.16).
#   frg_it_p150_runs.sh PASS         benchmark rows for ARMS in order (default "base A B C"; give "C B A base" for the
#                                    reversed pass): frgap/p150_it_ring.yaml, torus/torusxy_2x4_sizes.yaml,
#                                    torus/torusxy_4x2_sizes.yaml (4x2 descriptor)
# Arms: base = off, A = T 1 K 2 (65794), B = T 1 K 6 (65798), C = T 4 K 6 (66566). One kernel cache per arm.
# Labels frgit<PASS>_<arm>_{ring,t24,t42} (val: frgitv_<arm>_{ring,t24,t42}). A hang stops the job (the lb2 job then
# captures, resets and checks the links). An arm that fails validation (data mismatch, no hang) is listed in
# $out/frgit_failed_arms and left out of the benchmark passes.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS \
    TT_FABRIC_EXP_DRISC_DELAY TT_MESH_GRAPH_DESC_PATH
P=$1
[ -n "$P" ] || { echo "usage: [ARMS=...] frg_it_p150_runs.sh val|PASS"; exit 1; }
MGD=TT_MESH_GRAPH_DESC_PATH=$PWD/fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto
FAILED=$out/frgit_failed_arms
for f in $CON/frgap/p150_it_ring.yaml fabric_exp/v4/val_ring_p150.yaml fabric_exp/torus/val_torus_xy_p150.yaml \
         fabric_exp/torus/torusxy_2x4_sizes.yaml fabric_exp/torus/torusxy_4x2_sizes.yaml ${MGD#*=}; do
    [ -f $f ] || { echo "no $f"; exit 1; }
done
arm_env() {
    case $1 in
        base) echo "" ;;
        A) echo TT_FABRIC_EXP_DRISC_DELAY=65794 ;;
        B) echo TT_FABRIC_EXP_DRISC_DELAY=65798 ;;
        C) echo TT_FABRIC_EXP_DRISC_DELAY=66566 ;;
        *) echo "!!! unknown arm $1" >&2; return 1 ;;
    esac
}
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
python3 $CON/swap.py apply fabric_exp/variants/v4rxg3it $BK || stop
one() {  # label yaml arm [ENV...]
    local label=$1 y=$2 a=$3 e rc
    shift 3
    e=$(arm_env $a) || stop
    run_cap $label $y /home/aagarwal/.tt-metal-cache-frgit-$a 600 TT_FABRIC_EXP_LOG_SLOTS=1 $e "$@"; rc=$?
    [ $rc = 2 ] && { echo "!!! $label: hang (captured): stopping"; stop; }
    # No result line: the process crashed or was killed (job 170018 on a08u28: a SIGKILL, then UMD aborts on a device
    # that lost its links and PCIe). Stop, so the lb2 job resets and checks the links; never run on into the next arm.
    if [ $rc != 0 ] && ! sed 's/\x1b\[[0-9;]*m//g' $out/$label.log | grep -aqE 'Tests: [0-9]+ executed'; then
        echo "!!! $label: no result (crash or kill, not a data check): stopping"; stop
    fi
    return $rc
}
if [ "$P" = val ]; then
    rm -f $FAILED
    for a in ${ARMS:-base A B C}; do
        echo "##### validation, arm $a (job at ${SECONDS} s)"
        ok=1
        one frgitv_${a}_ring fabric_exp/v4/val_ring_p150.yaml $a || ok=0
        one frgitv_${a}_t24 fabric_exp/torus/val_torus_xy_p150.yaml $a || ok=0
        one frgitv_${a}_t42 fabric_exp/torus/val_torus_xy_p150.yaml $a $MGD || ok=0
        [ $ok = 1 ] || { echo "!!! arm $a failed validation: left out of the benchmark passes"; echo $a >> $FAILED; }
    done
else
    for a in ${ARMS:-base A B C}; do
        if [ -f $FAILED ] && grep -qx $a $FAILED; then echo "skipped arm $a (failed validation)"; continue; fi
        echo "##### pass $P, arm $a (job at ${SECONDS} s)"
        one frgit${P}_${a}_ring $CON/frgap/p150_it_ring.yaml $a || stop
        one frgit${P}_${a}_t24 fabric_exp/torus/torusxy_2x4_sizes.yaml $a || stop
        one frgit${P}_${a}_t42 fabric_exp/torus/torusxy_4x2_sizes.yaml $a $MGD || stop
    done
fi
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
echo ALL_DONE
