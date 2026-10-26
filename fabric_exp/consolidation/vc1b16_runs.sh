#!/bin/bash
# VC1 with batching only on rings of 16+ slots (RESULTS §98): v3q16 (v3q with rings under 16 slots credited slot by
# slot; offline it differs from v3q only on the 48 VC1 E1 configs) against v3q (= v3zy's code) and v3z, interleaved, at
# the builder's level (auto = -Os), -O2 and -O3, two rounds, order rotating; hang capture armed.
# Labels p37<v>r<i>vc1_<level> (p35: the a08u24 attempt, job 361, link training timeout).
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
V=fabric_exp/variants
for i in 1 2; do
    if [ $i = 1 ]; then order="v3q16 v3q v3z"; else order="v3z v3q v3q16"; fi
    for v in $order; do
        VC1_RUN=run_vc1_cap vc1_variant $v $V/$v p37${v#v3}r${i}vc1 auto O2 O3
    done
done
rm -f $M
echo ALL_DONE
