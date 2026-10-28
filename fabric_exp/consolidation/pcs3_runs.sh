#!/bin/bash
# Phase/cost analysis, stage 3 (RESULTS §98): v3q at -O3, PC sampling of every router ERISC while one long row runs.
#   pcs3_runs.sh 2D | 1D
# A fresh kernel cache per topology: the first (short) row compiles the routers and logs their compile commands, from
# which pcs_sections.py rebuilds them with -g for source mapping; every later row of that topology reuses the builds.
# Per row: fabric_exp/pcs3/<row>.yaml, label pcs3_<row>_13129088_trial (rate), samples results/pcs3_<row>_samples.*.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
out=fabric_exp/results
case "$1" in
2D) C=/home/aagarwal/.tt-metal-cache-pcs3-2d; ROWS="ew4352 ew16 ns4352 ewscatter ewfused ewatomic hop3 mcast fwd bidir"; FIRST=compile2d ;;
1D) C=/home/aagarwal/.tt-metal-cache-pcs3-1d; ROWS="lin1 lin3 linmcast"; FIRST=compile1d ;;
*) echo "usage: pcs3_runs.sh 2D|1D"; exit 1 ;;
esac
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $V/v3q $BK || { echo STOPPED; exit 1; }
run pcs3_${FIRST}_13129088_trial fabric_exp/pcs3/$FIRST.yaml $C O3 || { echo STOPPED; exit 1; }
for r in $ROWS; do
    lab=pcs3_${r}_13129088_trial
    rm -f $out/$lab.log $out/pcs3_${r}_samples.bin $out/pcs3_${r}_samples.json
    ( run $lab fabric_exp/pcs3/$r.yaml $C O3 ) & RP=$!
    timeout 300 python_env/bin/python $CON/pc_sampler.py $out/pcs3_${r}_samples --start-log $out/$lab.log \
        --max-s 120 2>&1 | grep --line-buffered -E '^sampler: [0-9]+ samples|Error|error|Traceback'
    wait $RP || { echo "!!! $r failed"; break; }
    awk -F, -v r=$r 'NR>1 {printf "%s: %s %s %.2f cycles/packet\n", r, $1, $3, 1.35e9/$10}' $out/$lab.csv
done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
