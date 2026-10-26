#!/bin/bash
# VC1 multicast attribution (RESULTS §98): v3z's multicast row is 13 to 14 behind v3x's on every repeat (job 352), but
# v3yb's E0 left multicast unchanged on v3v6's E1. Which of v3yb's two E0 changes interacts with v3x's E1?
#   v3zy = v3y's E0 (per-round work out of line, helpers) + v3x3's E1
#   v3zc = v3yc's E0 (helpers, completions before the send decision) + v3x3's E1
# with v3z and v3x as same-session references; -O3, two rounds, order rotating. Labels p33<v>r<i>vc1_o3.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
V=fabric_exp/variants
for i in 1 2; do
    if [ $i = 1 ]; then order="zy zc z x"; else order="x z zc zy"; fi
    for d in $order; do
        VC1_RUN=run_vc1_cap vc1_variant v3$d $V/v3$d p33${d}r${i}vc1 O3
    done
done
rm -f $M
echo ALL_DONE
