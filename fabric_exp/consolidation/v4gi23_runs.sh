#!/bin/bash
# The fixes for v4gi's losses (job 093; the burst profile, job 094, showed more and smaller completion credits after
# partial credits, and slots held longer: the rest of a group was due only once more than a whole group waited). Both
# add a READ bit to the slot records (no proof for a group no command read: atomics, inline writes):
#   v4gi2: the due unit is the rest of the oldest group (credits realign at once after a partial credit);
#   v4gi3: v4vc's unit, the oldest T slots, spanning at most two groups (two ID proofs when it does).
# Against v4gi (label v4gic) and v4vc (v4vcd) in the same job:
#   1. the VC1 split suite at -O3, hang capture armed (labels <label>_o3);
#   2. VC0-only builds: data-checked validation, then every suite and the latency rows, as v4x_runs.sh.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source $CON/vc1_lib.sh
source $CON/vc1_capture_lib.sh
for vl in v4gi2:v4gi2 v4gi3:v4gi3 v4gi:v4gic v4vc:v4vcd; do
    VC1_RUN=run_vc1_cap vc1_variant ${vl##*:} fabric_exp/variants/${vl%%:*} ${vl##*:} O3
done
bash $CON/v4x_runs.sh v4gi2 v4gi3 v4gi:v4gic v4vc:v4vcd || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
