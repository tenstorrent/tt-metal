#!/bin/bash
# Bubble-threshold sweep (variants/v4txa11t/README.md), P150: v4txa11t swapped in once; arm TN sets
# TT_FABRIC_EXP_DRISC_DELAY = N, the free far slots an injection channel on a deadlock-avoiding VC needs to send
# (T2 = unset = the switch off = v4txa11's images).
#   frg_bt_p150_runs.sh val          data-checked validation per arm (ARMS, default the on-arms "T3 T4 T5 T6"), only the
#                                    rows proven on the LoudBoxes: v4/val_ring_p150.yaml, torus/val_torus_xy_p150.yaml on
#                                    the 2x4 and the 4x2 (fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto) XY torus.
#                                    No flow-control full-ring row on a LoudBox (FULLRING_MCAST_GAP.md §8.16).
#   frg_bt_p150_runs.sh PASS         benchmark rows for ARMS in order (default "T2 T3 T4 T5 T6"; give "T6 T5 T4 T3 T2"
#                                    for the reversed pass): frgap/p150_it_ring.yaml, torus/torusxy_2x4_sizes.yaml,
#                                    torus/torusxy_4x2_sizes.yaml (4x2 descriptor)
# Rule-cost arms (RING_JAM_FIX.md §9.4; TAG=frgbu): A2 = today (T2 = T2), A16 = T 16, C2 = the ring routers' slots
# 16/51/16 (TT_FABRIC_EXP_SLOTS_1D) at T 2, T8 = T 8 (the torus receivers' half). Each arm runs only its rows
# (ARM_ROWS): A2 ring + torus, A16 and C2 ring, T8 torus; validation covers the arm's rows (A2 skips it).
# Fraction-vs-cap arms (RING_JAM_FIX.md §9.5; TAG=frgbv): A2, A16, C2 as above, C8 = 16/51/16 at T 8, G2 / G32 / G48 =
# the ring routers' slots 18/17/64 at T 2 / 32 / 48. Rows: ring (p150_it_ring.yaml) for A2, A16, C2, C8; ring35
# (p150_it_ring_mps3584.yaml, a 3584 B router max packet, so a 64-slot receiver fits) for every arm. Validation per
# row kind: val_ring_p150.yaml (ring), val_ring_p150_mps3584.yaml (ring35).
# One kernel cache per arm. Labels <TAG><PASS>_<arm>_{ring,t24,t42} (val: <TAG>v_<arm>_...), TAG default frgbt. A hang stops the job (the
# lb2 job then captures, resets and checks the links). An arm that fails validation (data mismatch, no hang) is listed in
# $out/frgbt_failed_arms and left out of the benchmark passes; a run with no result line (crash, kill) stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS \
    TT_FABRIC_EXP_DRISC_DELAY TT_MESH_GRAPH_DESC_PATH TT_FABRIC_PHASE_PROFILE TT_FABRIC_PHASE_PROFILE_CSV
P=$1
[ -n "$P" ] || { echo "usage: [ARMS=...] frg_bt_p150_runs.sh val|PASS"; exit 1; }
MGD=TT_MESH_GRAPH_DESC_PATH=$PWD/fabric_exp/torus/p150_x8_4x2_torus_xy_mgd.textproto
FAILED=$out/${TAG:-frgbt}_failed_arms
for f in $CON/frgap/p150_it_ring.yaml fabric_exp/v4/val_ring_p150.yaml fabric_exp/torus/val_torus_xy_p150.yaml \
         $CON/frgap/p150_it_ring_mps3584.yaml $CON/frgap/val_ring_p150_mps3584.yaml \
         fabric_exp/torus/torusxy_2x4_sizes.yaml fabric_exp/torus/torusxy_4x2_sizes.yaml ${MGD#*=}; do
    [ -f $f ] || { echo "no $f"; exit 1; }
done
TAG=${TAG:-frgbt}
arm_env() {
    case $1 in
        T2 | A2) echo "" ;;
        T[3-9] | T1[0-6]) echo TT_FABRIC_EXP_DRISC_DELAY=${1#T} ;;
        A16) echo TT_FABRIC_EXP_DRISC_DELAY=16 ;;
        C2) echo TT_FABRIC_EXP_SLOTS_1D=16,51,16 ;;
        C8) echo TT_FABRIC_EXP_SLOTS_1D=16,51,16 TT_FABRIC_EXP_DRISC_DELAY=8 ;;
        G2) echo TT_FABRIC_EXP_SLOTS_1D=18,17,64 ;;
        G32 | G48) echo TT_FABRIC_EXP_SLOTS_1D=18,17,64 TT_FABRIC_EXP_DRISC_DELAY=${1#G} ;;
        *) echo "!!! unknown arm $1" >&2; return 1 ;;
    esac
}
arm_rows() {  # the rows an arm runs (default every row)
    case $TAG:$1 in
        frgbu:A16 | frgbu:C2) echo ring ;;
        frgbu:T8) echo "t24 t42" ;;
        frgbv:G*) echo ring35 ;;
        frgbv:*) echo "ring ring35" ;;
        *) echo "ring t24 t42" ;;
    esac
}
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
python3 $CON/swap.py apply fabric_exp/variants/v4txa11t $BK || stop
one() {  # label yaml arm [ENV...]
    local label=$1 y=$2 a=$3 e rc
    shift 3
    e=$(arm_env $a) || stop
    run_cap $label $y /home/aagarwal/.tt-metal-cache-$TAG-$a 600 TT_FABRIC_EXP_LOG_SLOTS=1 $e "$@"; rc=$?
    [ $rc = 2 ] && { echo "!!! $label: hang (captured): stopping"; stop; }
    if [ $rc != 0 ] && ! sed 's/\x1b\[[0-9;]*m//g' $out/$label.log | grep -aqE 'Tests: [0-9]+ executed'; then
        echo "!!! $label: no result (crash or kill, not a data check): stopping"; stop
    fi
    return $rc
}
if [ "$P" = val ]; then
    rm -f $FAILED
    for a in ${ARMS:-T3 T4 T5 T6}; do
        echo "##### validation, arm $a (job at ${SECONDS} s)"
        ok=1
        rows=" $(arm_rows $a) "
        [[ $rows == *" ring "* ]] && { one ${TAG}v_${a}_ring fabric_exp/v4/val_ring_p150.yaml $a || ok=0; }
        [[ $rows == *" ring35 "* ]] && { one ${TAG}v_${a}_ring35 $CON/frgap/val_ring_p150_mps3584.yaml $a || ok=0; }
        [[ $rows == *" t24 "* ]] && { one ${TAG}v_${a}_t24 fabric_exp/torus/val_torus_xy_p150.yaml $a || ok=0; }
        [[ $rows == *" t42 "* ]] && { one ${TAG}v_${a}_t42 fabric_exp/torus/val_torus_xy_p150.yaml $a $MGD || ok=0; }
        [ $ok = 1 ] || { echo "!!! arm $a failed validation: left out of the benchmark passes"; echo $a >> $FAILED; }
    done
else
    for a in ${ARMS:-T2 T3 T4 T5 T6}; do
        if [ -f $FAILED ] && grep -qx $a $FAILED; then echo "skipped arm $a (failed validation)"; continue; fi
        echo "##### pass $P, arm $a (job at ${SECONDS} s)"
        rows=" $(arm_rows $a) "
        [[ $rows == *" ring "* ]] && { one ${TAG}${P}_${a}_ring $CON/frgap/p150_it_ring.yaml $a || stop; }
        [[ $rows == *" ring35 "* ]] && { one ${TAG}${P}_${a}_ring35 $CON/frgap/p150_it_ring_mps3584.yaml $a || stop; }
        [[ $rows == *" t24 "* ]] && { one ${TAG}${P}_${a}_t24 fabric_exp/torus/torusxy_2x4_sizes.yaml $a || stop; }
        [[ $rows == *" t42 "* ]] && { one ${TAG}${P}_${a}_t42 fabric_exp/torus/torusxy_4x2_sizes.yaml $a $MGD || stop; }
    done
fi
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
echo ALL_DONE
