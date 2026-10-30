#!/bin/bash
# Data-checked validation of one router variant across the feature configs (2026-10-29; for structural changes such as
# the sender cleanup, V4_SENDER_SHAPE.md): variant_val_runs.sh VARIANT. Hang capture armed; a hang stops the job;
# failures are listed at the end. Runs, labels val_<variant>_<run>:
#   two ERISCs: mesh sanity (mesh), val1d_m1 (val1d), ring and Y-torus validation (ring, torusy), mux mode (mux), the
#   VC2 yaml (vc2), NeighborExchange (nbr), main's channel-trimming gtests (trim);
#   one ERISC: mesh sanity, ring validation, VC2, NeighborExchange, trimming gtests (*_1e).
# The VC1 split is validated by census_vc1_runs.sh (vc1_alloc.yaml, data-checked, with its own hang capture).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
V=$1
[ -n "$V" ] && [ -d fabric_exp/variants/$V ] || { echo "usage: variant_val_runs.sh VARIANT"; exit 1; }
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
E="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
VC2Y=tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_vc2_at_least_2x2_mesh.yaml
C=/home/aagarwal/.tt-metal-cache-val-$V
rm -rf "${C:?}-two" "${C:?}-one"; mkdir -p $C-two $C-one
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
FAILED=""
v() {  # run yaml cache timeout [ENV...]
    local r=$1 y=$2 c=$3 t=$4; shift 4
    run_cap val_${V}_$r "$y" $c $t "$@"
    case $? in 2) stop ;; 1) FAILED="$FAILED $r" ;; esac
}
python3 $CON/swap.py apply fabric_exp/variants/$V $BK || stop
echo "##### two ERISCs (job at ${SECONDS} s)"
v mesh $(suite_yaml mesh) $C-two 300
v val1d $(suite_yaml val1d_m1) $C-two 300
v ring fabric_exp/v4/val_ring_p150.yaml $C-two 120
v torusy fabric_exp/v4/val_torus_y_p150.yaml $C-two 120
v mux fabric_exp/v4/val_mux.yaml $C-two 300
v vc2 $VC2Y $C-two 600
v nbr fabric_exp/v4/val_nbr.yaml $C-two 300
v trim "gtest:ChannelTrimming*" $C-two 1300
echo "##### one ERISC (job at ${SECONDS} s)"
v mesh_1e $(suite_yaml mesh) $C-one 300 $E
v ring_1e fabric_exp/v4/val_ring_p150.yaml $C-one 120 $E
v vc2_1e $VC2Y $C-one 600 $E
v nbr_1e fabric_exp/v4/val_nbr.yaml $C-one 300 $E
v trim_1e "gtest:ChannelTrimming*" $C-one 1300 $E
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo "failed:${FAILED:- none}"
echo "ALL_DONE (job at ${SECONDS} s)"
