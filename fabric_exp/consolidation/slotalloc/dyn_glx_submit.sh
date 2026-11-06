#!/bin/bash
# Submit one pass of the dynamic slot allocator's Galaxy A/B (dyn_glx.sbatch; SLOT_ALLOCATOR.md §8), after counting
# the study's Galaxy allocations: at most 4 running + pending (user, 2026-11-02), counted with squeue -u aagarwal on
# the bh_sc36_6* and bh_110a_sc16 partitions, the multi-Galaxy quad job (glxq) excepted (user, 2026-11-03). Pass b reverses the arms
# within each kind. Expected rows: slot_alloc_table.sh, 2026-11-03.
#   dyn_glx_submit.sh a|b [--test-only]        (TREE=v4txa11_dyn unless set)
# Excluded nodes (coordinator, 2026-11-03): bh-glx-110-a01u14 (/tmp full, Open MPI dies at init), a02u02 and a02u08
# (earlier foreign processes). a02u14 can't map rings: excluded only for ring/torus rows, which these passes don't have
# (EXCLUDE overrides the list).
set -euo pipefail
# a03u14 and a08u20 stall multi-process tt-run (coordinator, 2026-11-04): excluded, since the VC1 kinds run two ranks.
EXCLUDE=${EXCLUDE:-bh-glx-110-a01u14,bh-glx-110-a02u02,bh-glx-110-a02u08,bh-glx-110-a03u14,bh-glx-110-a08u20}
# Partitions (user, 2026-11-04): bh_sc36_6 and bh_110a_sc16 (bh-glx-110-a07..a10, 6 h limit; the job asks 1h20).
PARTS=bh_sc36_6,bh_110a_sc16
PASS=${1:?usage: dyn_glx_submit.sh a|b [--test-only]}
MODE=${2:-}   # read before the step loop, whose `set --` replaces the positional arguments
cd /home/aagarwal/tt-metal
KINDS=(
  "1d tier=16,16,32 dyn=26,25,32 dynw24=24,27,32"
  "vc1 tier=51,4,8,2,4 dyn=19,9,16,5,4"
  "vc2k tier=43,7,11+2,4 def=17,14,16+2,4 dyn=17,14,16+4,2"
  "2d3k tier=35,16,32 dyn=27,24,16 dynw24=24,25,16 row=20,21,32"
  "vc13k tier=85,4,8,2,4 dyn=30,14,16,7,6 dynw24=24,16,16,7,6 row=20,15,16,9,7"
  "vc23k tier=77,7,11+2,4 def=51,14,16+2,4 dyn=24,22,16+5,4 row=20,14,16+16,16"
)
# A tree whose host library has the VC0 receiver floor (2026-11-04) adds the floor's 3072 B rows (dynfloor) next to
# the floor-off dyn rows; on the first A/B's tree (v4txa11_dyn) the passes stay as they were.
if grep -aq vc0_receiver_floor /data/aagarwal/frs-glx/trees/${TREE:-v4txa11_dyn}/build_Release/tt_metal/libtt_metal.so; then
    KINDS[3]+=" dynfloor=23,20,32"
    KINDS[4]+=" dynfloor=26,10,32,7,6"
    echo "tree ${TREE:-v4txa11_dyn}: host with the VC0 receiver floor; dynfloor arms added"
fi
STEPS=""
for k in "${KINDS[@]}"; do
    set -- $k; kind=$1; shift
    arms=("$@")
    [ $PASS = b ] && arms=($(printf '%s\n' "${arms[@]}" | tac))
    for a in "${arms[@]}"; do STEPS+="$kind:$PASS:$a "; done
done
n=$(squeue -u aagarwal -h -o "%P %j" | awk '$1 ~ /bh_sc36_6|bh_110a_sc16/ && $2 !~ /^glxq/' | wc -l)
echo "study Galaxy allocations now (running + pending, quad excepted): $n of 4"
echo "STEPS=$STEPS"
echo "partitions: $PARTS; excluded: $EXCLUDE"
[ "$MODE" = --test-only ] && { STEPS="$STEPS" TREE=${TREE:-v4txa11_dyn} sbatch --test-only --partition=$PARTS --exclude=$EXCLUDE fabric_exp/consolidation/slotalloc/dyn_glx.sbatch; exit 0; }
[ $n -lt 4 ] || { echo "!!! the cap of 4 is reached: not submitting"; exit 1; }
STEPS="$STEPS" TREE=${TREE:-v4txa11_dyn} sbatch --partition=$PARTS --exclude=$EXCLUDE fabric_exp/consolidation/slotalloc/dyn_glx.sbatch
