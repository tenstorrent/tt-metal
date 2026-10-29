#!/bin/bash
# VC2 hang isolation (2026-10-29; RESULTS §98 "VC2 on v4: v4vc2, first device runs": job 142 hung in
# VC2MeshNeighbourExchangeVC2, every VC2 flow stopped at its 512-packet flow-control window). User: fix the capture
# first (hang_capture.py / hang_workers.py now read the post-merge go rings, so the test kernels are captured), then
# run the VC2 groups one at a time, smallest first, and stop at the first hang. v4vc2, two ERISCs, the study's settings.
# Groups (fabric_exp/vc2/<group>.yaml, split from test_fabric_vc2_at_least_2x2_mesh.yaml, each vetted on the mock):
#   1. MeshNeighbourExchangeVC2Unidirectional   one VC2 flow, chip 0 -> chip 1
#   2. VC2MeshNeighbourExchangeVC2Bidirectional two VC2 flows between chips 0 and 1
#   3. MeshNeighbourExchangeVC0VC2Unidirectional a VC2 and a VC0 flow, chip 0 -> chip 1
#   4. VC2MeshNeighbourExchangeVC2              the group that hung (VC2 and VC0 neighbour exchange, every chip)
# Logs fabric_exp/results/vc2iso_<n>.log; a hang stops the job (capture in results/vc2iso_<n>_hangcap).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/cap_lib.sh
[ -n "$out" ] && [ -d "$out" ] || { echo "no results directory"; exit 1; }
unset TT_FABRIC_EXP_SLOTS_VC1 TT_FABRIC_EXP_SLOTS TT_FABRIC_EXP_SLOTS_2D TT_FABRIC_EXP_SLOTS_1D TT_FABRIC_EXP_TIER_SLOTS
c=/home/aagarwal/.tt-metal-cache-vc2iso
rm -rf "${c:?}"; mkdir -p $c
python3 $CON/swap.py apply fabric_exp/variants/v4vc2 $BK || { echo STOPPED; exit 1; }
n=0
for g in MeshNeighbourExchangeVC2Unidirectional VC2MeshNeighbourExchangeVC2Bidirectional \
    MeshNeighbourExchangeVC0VC2Unidirectional VC2MeshNeighbourExchangeVC2; do
    n=$((n + 1))
    run_cap vc2iso_$n fabric_exp/vc2/$g.yaml $c 240
    if [ $? = 2 ]; then
        python3 $CON/swap.py restore $BK
        echo "STOPPED at group $n ($g)"
        exit 1
    fi
done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
