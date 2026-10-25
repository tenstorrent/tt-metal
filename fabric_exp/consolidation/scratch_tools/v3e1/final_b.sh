#!/bin/bash
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3e1
cd /home/aagarwal/tt-metal
rm -rf $S/b1_* $S/probeb $S/pbr_hook $S/pbr_statefield
$S/build_all.sh fabric_exp/variants/v3e1b $S/b1
python3 fabric_exp/consolidation/perturb.py fabric_exp/variants/v3e1b $S/probeb hook statefield
for p in hook statefield; do python3 $S/ob.py build --variant $S/probeb/$p --out $S/pbr_$p --opt O3 --jobs 32 2>&1 | tail -1; done
echo FINAL_DONE
