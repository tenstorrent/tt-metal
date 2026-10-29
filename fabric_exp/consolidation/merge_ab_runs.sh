#!/bin/bash
# Before / after a merge of main (2026-10-28: main brings sfpi 7.86.0 -> 7.91.0, firmware, NoC API, test-infra and
# main's router changes). The same runs on each side, depths logged (TT_FABRIC_EXP_LOG_SLOTS=1), fresh caches:
#   merge_ab_runs.sh TAG a   v4gi: validation (mesh, 1D, ring, Y-torus) then every benchmark suite as v4x_runs.sh
#                            (4352, mix, flushmix, flushmix4, burst, 1D, latency, ring, Y-torus);
#                            labels TAG_v4gi_13129088_<suite>, TAG_v4gi_val_{ring,torusy}_p150
#   merge_ab_runs.sh TAG b   v4gi on the VC1 split (vc1_alloc.yaml with hang capture, vc1_inter.yaml; labels
#                            TAGvc1_v4gi_default_p1_o3, TAGvc1i_v4gi_default_p1_o3), then main's router as shipped
#                            (kernel v1, flags 0, firmware keepalive): mesh sanity, 2D 4352, 1D (exp_1d_4352.yaml)
#                            at -O3, and the VC1 split at the builder's level (labels TAG_v1_*, TAGvc1_v1_default_p1_auto)
# TAG: mpre (today's tree, sfpi 7.86.0) or mpost (after the merge). VARIANT (default v4gi) picks the variant
# directory swapped in for the v4gi labels: after the merge v4gi_m (v4gi with main's changes to the files it overlays).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
T=$1; part=$2
[ -n "$T" ] && { [ "$part" = a ] || [ "$part" = b ]; } || { echo "usage: merge_ab_runs.sh TAG a|b"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
export TT_FABRIC_EXP_LOG_SLOTS=1
EXTRA_ENV="TT_FABRIC_EXP_LOG_SLOTS=1"
echo "variant ${VARIANT:-v4gi}; sfpi: $(runtime/sfpi/compiler/bin/riscv-tt-elf-g++ --version | head -1); tree: $(git rev-parse --short HEAD)"
if [ $part = a ]; then
    C=/home/aagarwal/.tt-metal-cache-$T-v4gi
    rm -rf "${C:?}"; mkdir -p $C
    python3 $CON/swap.py apply fabric_exp/variants/${VARIANT:-v4gi} $BK || { echo STOPPED; exit 1; }
    run_cap ${T}_v4gi_13129088_mesh $(suite_yaml mesh) $C 300 || { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
    run_cap ${T}_v4gi_13129088_val1d $(suite_yaml val1d_m1) $C 300 || { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
    run_cap ${T}_v4gi_val_ring_p150 fabric_exp/v4/val_ring_p150.yaml $C 120 || { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
    run_cap ${T}_v4gi_val_torusy_p150 fabric_exp/v4/val_torus_y_p150.yaml $C 120 || { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
    for s in 4352 mix flushmix flushmix4 burst 1d_m1 lat; do
        run_cap ${T}_v4gi_13129088_$s $(suite_yaml $s) $C 200; [ $? = 2 ] && { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
    done
    run_cap ${T}_v4gi_13129088_ring fabric_exp/v4/exp_ring.yaml $C 120; [ $? = 2 ] && { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
    run_cap ${T}_v4gi_13129088_torusy fabric_exp/v4/exp_torus_y.yaml $C 120; [ $? = 2 ] && { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
else
    c=/home/aagarwal/.tt-metal-cache-p1vc1-$T
    python3 $CON/swap.py apply fabric_exp/variants/${VARIANT:-v4gi} $BK || { echo STOPPED; exit 1; }
    VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 run_vc1_cap ${T}vc1_v4gi_default_p1_o3 $c O3 \
        || { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
    VC1_YAML=fabric_exp/vc1/vc1_inter.yaml VC1_TIMEOUT=180 run_vc1 ${T}vc1i_v4gi_default_p1_o3 $c O3 || echo "!!! inter rows failed"
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
    MAIN="TT_FABRIC_KERNEL_VERSION=1 TT_FABRIC_EXP_FLAGS=0"
    C=/home/aagarwal/.tt-metal-cache-$T-v1
    rm -rf "${C:?}"; mkdir -p $C
    KA_ENV= run_cap ${T}_v1_13129088_mesh $(suite_yaml mesh) $C 300 $MAIN || { echo STOPPED; exit 1; }
    for s in 4352 1d; do
        KA_ENV= run_cap ${T}_v1_13129088_$s $(suite_yaml $s) $C 200 $MAIN; [ $? = 2 ] && { echo STOPPED; exit 1; }
    done
    KA_ENV= VC1CAP_EXTRA_ENV="$MAIN" VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 \
        run_vc1_cap ${T}vc1_v1_default_p1_auto $c auto || { echo STOPPED; exit 1; }
fi
rm -f $M
echo ALL_DONE
