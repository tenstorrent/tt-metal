#!/bin/bash
# PC-sampling trial, stage 2 (RESULTS §98): v3q at -O3 on fabric_exp/pcs_trial.yaml (one long 2D 1-hop E/W row).
#   A: no sampler (first run compiles; baseline rate)
#   B: the same run with pc_sampler.py reading every router ERISC's PC over the debug bus while it runs
#   C: no sampler again (the device still behaves; second baseline)
# Checks: B's rate equals A's and C's (no perturbation), the sampler opens and reads without errors, PCs look sane.
# Labels pcsA/pcsB/pcsC_13129088_trial; samples in fabric_exp/results/pcsB_samples.{bin,json}.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
C=/home/aagarwal/.tt-metal-cache-pcs
Y=fabric_exp/pcs_trial.yaml
out=fabric_exp/results
mkdir -p $C
python3 $CON/swap.py apply $V/v3q $BK || { echo STOPPED; exit 1; }
run pcsA_13129088_trial $Y $C O3 || { echo STOPPED; exit 1; }
rm -f $out/pcsB_13129088_trial.log $out/pcsB_samples.bin $out/pcsB_samples.json
( run pcsB_13129088_trial $Y $C O3 ) & RP=$!
timeout 240 python_env/bin/python $CON/pc_sampler.py $out/pcsB_samples --start-log $out/pcsB_13129088_trial.log \
    --max-s 120 2>&1 | grep --line-buffered -E '^sampler:|Error|error|Traceback'
echo "sampler exit ${PIPESTATUS[0]}"
wait $RP || echo "!!! run B failed"
run pcsC_13129088_trial $Y $C O3 || { echo STOPPED; exit 1; }
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
for l in pcsA pcsB pcsC; do
    awk -F, -v l=$l 'NR>1 {printf "%s %s %s %.2f cycles/packet\n", l, $1, $3, 1.35e9/$10}' $out/${l}_13129088_trial.csv
done
rm -f $M
echo ALL_DONE
