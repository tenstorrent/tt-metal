#!/bin/bash
# Which ring and torus tests this P150x8 runs: every test of val_ring, val_torus_y, exp_ring, exp_torus_y on its own
# (fabric_exp/v4/probe/<set>_<name>.yaml) on the tree's own router (main, kernel v1). Hang capture armed.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source /home/aagarwal/tt-metal/fabric_exp/consolidation/cap_lib.sh
C=/home/aagarwal/.tt-metal-cache-v1-ringval
for n in $(cat fabric_exp/v4/probe/list.txt); do
    [ $SECONDS -gt 1200 ] && { echo "budget: stopping before $n"; break; }
    run_cap v1p_$n fabric_exp/v4/probe/$n.yaml $C 150 TT_FABRIC_KERNEL_VERSION=1 > fabric_exp/results/probe_tmp.txt 2>&1
    rc=$?
    why=$(sed 's/\x1b\[[0-9;]*m//g' fabric_exp/results/v1p_$n.log | grep -aoE 'TT_(FATAL|THROW): [^(]*' | head -1 | cut -c1-110)
    printf "%-4s %-45s %s\n" "$rc" "$n" "$(grep -oE 'Tests: .*failed|NO RESULT' fabric_exp/results/probe_tmp.txt | head -1) $why"
    [ $rc = 2 ] && { cat fabric_exp/results/probe_tmp.txt; break; }
done
rm -f fabric_exp/results/probe_tmp.txt $M
