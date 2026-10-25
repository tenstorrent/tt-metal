#!/bin/bash
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3e1
cd /home/aagarwal/tt-metal
rm -rf $S/r5_* $S/probe $S/pr_hook $S/pr_statefield $S/prb_statefield
$S/build_all.sh fabric_exp/variants/v3e1 $S/r5
python3 fabric_exp/consolidation/perturb.py fabric_exp/variants/v3e1 $S/probe hook statefield
for p in hook statefield; do python3 $S/ob.py build --variant $S/probe/$p --out $S/pr_$p --opt O3 --jobs 32 2>&1 | tail -1; done
python3 $S/ob.py build --variant $S/probe_base/statefield --out $S/prb_statefield --opt O3 --jobs 32 2>&1 | tail -1
echo FINAL_DONE
