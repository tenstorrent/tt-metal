#!/bin/bash
# VC1 multicast repeat (RESULTS §98): v3z against v3x, interleaved, at the builder's level (auto = -Os) and -O3. With
# one run each, v3z's multicast row was 12 to 14 behind v3x's while the two meshes differed by 8 to 15 within a run.
# Four pairs per level, the order alternating between pairs; hang capture armed. Labels p32<z|x>r<i>vc1_<level>.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
V=fabric_exp/variants
for i in 1 2 3 4; do
    if [ $((i % 2)) = 1 ]; then order="z x"; else order="x z"; fi
    for d in $order; do
        VC1_RUN=run_vc1_cap vc1_variant v3$d $V/v3$d p32${d}r${i}vc1 auto O3
    done
done
rm -f $M
echo ALL_DONE
