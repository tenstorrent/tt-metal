#!/bin/bash
# VC1 + VC2 together (2026-11-01, user: a test; V4_RECEIVER_SHAPE.md §28): the VC1 split (two 2x2 meshes, one MPI rank
# each, vc1_lib.sh) with VC2 on, router VARIANT (default v4rxc7v: v4rxc7 with VC2 admitted alongside VC1), slots logged
# (study rows: 19/8/16/4/4 + VC2 2/4). vc1vc2_runs.sh PASS [VARIANT]; labels vcvc2<PASS>_<variant>_<run>:
#   two ERISCs: vc1_alloc.yaml (data-checked VC0/VC1 traffic, VC2 compiled in and idle; hang capture armed), vc1_inter;
#   one ERISC: vc1_alloc.yaml (hang capture armed);
#   then VC2 traffic, last (never run in the two-mesh setup before): test_fabric_vc2_at_least_2x2_mesh.yaml (VC2
#   neighbour exchange in each mesh), two ERISCs and one ERISC.
# A hang stops the job (capture); other failures are listed at the end.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
P=$1
R=${2:-v4rxc7v}
[ -n "$P" ] && [ -d fabric_exp/variants/$R ] || { echo "usage: vc1vc2_runs.sh PASS [VARIANT]"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
stop() { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
VC2="TT_METAL_ENABLE_FABRIC_VC2=1 TT_FABRIC_EXP_LOG_SLOTS=1"
ONE="TT_METAL_DISABLE_FABRIC_TWO_ERISC=1"
VC2Y=tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_vc2_at_least_2x2_mesh.yaml
L=vcvc2${P}_${R}
C=/home/aagarwal/.tt-metal-cache-vcvc2-$P
FAILED=""
python3 $CON/swap.py apply fabric_exp/variants/$R $BK || stop
echo "##### two ERISCs (job at ${SECONDS} s)"
VC1CAP_EXTRA_ENV="$VC2" VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 \
    run_vc1_cap ${L}_alloc $C-two O3 || stop
VC1CAP_EXTRA_ENV="$VC2" VC1_YAML=fabric_exp/vc1/vc1_inter.yaml VC1_TIMEOUT=180 run_vc1 ${L}_inter $C-two O3 \
    || FAILED="$FAILED inter"
echo "##### one ERISC (job at ${SECONDS} s)"
VC1CAP_EXTRA_ENV="$VC2 $ONE" VC1CAP_SPLIT_YAML=fabric_exp/vc1/vc1_alloc.yaml VC1CAP_SPLIT_FINISHES=50 \
    run_vc1_cap ${L}_alloc_1e $C-one O3 || stop
echo "##### VC2 traffic (job at ${SECONDS} s)"
VC1CAP_EXTRA_ENV="$VC2" VC1_YAML=$VC2Y VC1_TIMEOUT=300 run_vc1 ${L}_vc2 $C-two O3 || FAILED="$FAILED vc2"
VC1CAP_EXTRA_ENV="$VC2 $ONE" VC1_YAML=$VC2Y VC1_TIMEOUT=300 run_vc1 ${L}_vc2_1e $C-one O3 || FAILED="$FAILED vc2_1e"
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo "failed:${FAILED:- none}"
echo "ALL_DONE (job at ${SECONDS} s)"
