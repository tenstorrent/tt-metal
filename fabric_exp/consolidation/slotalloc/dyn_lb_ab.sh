#!/bin/bash
# Dynamic slot allocator A/B on one P150 LB (SLOT_ALLOCATOR.md §8; an lb2 job: all arms on one host): both passes of
# dyn_p150_runs.sh, pass a in the order below and pass b reversed. Needs the host library with TT_FABRIC_EXP_DYN_SLOTS
# (host_rebuild_dynslots.sh on a09u32, then lb2_stage.sh full). Expected rows: slot_alloc_table.sh, default policy and
# worker_cap_slots=24 (dynw24), 2026-11-03.
#   dyn_lb_ab.sh [TAG]      labels dyn<TAG>a_* / dyn<TAG>b_*
#   dyn_lb_ab.sh TAG PART   one broker-sized part (a09u32 broker, timeout 1500; 2026-11-04): a1 = pass a, the 4352 B
#                           suites; a2 = pass a, the 3072 B suites; b1 = pass b (reversed), the 3072 B suites; b2 = pass b
#                           (reversed), the 4352 B suites. Each suite keeps all its arms, so every part has the
#                           tier / default reference arm; run a1 a2 b1 b2 in that order (DYN_BUDGET 1300 s per part).
# On a host library with the VC0 receiver floor (2026-11-04), dyn and dynw24 run with it off (the rows below), and the
# dynfloor arm adds the floor's 3072 B rows next to them (slot_alloc_table.sh, SLOT_ALLOCATOR §10).
C=/home/aagarwal/tt-metal/fabric_exp/consolidation/slotalloc
PART=${2:-}
case $PART in "") export DYN_BUDGET=99999 ;; a1|a2|b1|b2) export DYN_BUDGET=${DYN_BUDGET:-1300} ;;
    *) echo "usage: dyn_lb_ab.sh [TAG [a1|a2|b1|b2]]"; exit 1 ;; esac
STEPS="1d:tier=16,16,32 1d:dyn=26,25,32 1d:dynw24=24,27,32
ring:tier=16,16,32 ring:dyn=26,25,32
vc1:tier=51,4,8,2,4 vc1:dyn=19,9,16,5,4
vc2e:tier=43,7,11+2,4 vc2e:def=17,14,16+2,4 vc2e:dyn=17,14,16+4,2
vc2alone:tier=43,7,11+2,4 vc2alone:def=17,14,16+2,4 vc2alone:dyn=17,14,16+4,2
vc02ew:tier=43,7,11+2,4 vc02ew:def=17,14,16+2,4 vc02ew:dyn=17,14,16+4,2
2d3k:tier=35,16,32 2d3k:dyn=27,24,16 2d3k:dynw24=24,25,16 2d3k:row=20,21,32
1d3k:tier=32,32,32 1d3k:dyn=43,42,32 1d3k:dynw24=24,61,32
ring3k:tier=32,32,32 ring3k:dyn=43,42,32
torusy3k:tier=35,16,32 torusy3k:dyn=27,24,16
vc13k:tier=85,4,8,2,4 vc13k:dyn=30,14,16,7,6 vc13k:dynw24=24,16,16,7,6 vc13k:row=20,15,16,9,7"
if grep -aq vc0_receiver_floor /home/aagarwal/tt-metal/build_Release/tt_metal/libtt_metal.so; then
    STEPS=${STEPS/2d3k:row=20,21,32/2d3k:row=20,21,32 2d3k:dynfloor=23,20,32}
    STEPS=${STEPS/vc13k:row=20,15,16,9,7/vc13k:row=20,15,16,9,7 vc13k:dynfloor=26,10,32,7,6}
fi
A=$(echo $STEPS)
B=$(echo $STEPS | tr ' ' '\n' | tac | tr '\n' ' ')
if [ -n "$PART" ]; then
    k4=(); k3=()   # 4352 B suites / 3072 B suites (names ending in 3k)
    case $PART in a*) list=$A ;; b*) list=$B ;; esac
    for st in $list; do case ${st%%:*} in *3k) k3+=($st) ;; *) k4+=($st) ;; esac; done
    case $PART in a1|b2) steps=("${k4[@]}") ;; a2|b1) steps=("${k3[@]}") ;; esac
    echo "part $PART: ${#steps[@]} steps"
    bash $C/dyn_p150_runs.sh ${1:-}${PART:0:1} "${steps[@]}" || exit 1
    echo "DYN_PART_${PART}_DONE"
    exit 0
fi
bash $C/dyn_p150_runs.sh ${1:-}a $A || exit 1
bash $C/dyn_p150_runs.sh ${1:-}b $B || exit 1
echo DYN_LB_AB_DONE
