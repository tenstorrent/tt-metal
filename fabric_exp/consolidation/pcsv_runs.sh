#!/bin/bash
# PC sampling of any router variant (RESULTS §98 method): pcsv_runs.sh VARIANT TAG 2D|1D|Ring|TorusY ROW...
# A fresh kernel cache: the first (short) row compiles the routers and logs their compile commands (for -g twins);
# each ROW (fabric_exp/pcs3/<row>.yaml) then runs with pc_sampler.py reading every router ERISC's PC.
# Labels pcsv_<tag>_<row>_13129088_trial (rate), samples results/pcsv_<tag>_<row>_samples.*.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
VAR=$1; TAG=$2; TOPO=$3; shift 3
out=fabric_exp/results
C=/home/aagarwal/.tt-metal-cache-pcsv-$TAG
case "$TOPO" in 2D) FIRST=compile2d ;; 1D) FIRST=compile1d ;; Ring) FIRST=compilering ;; TorusY) FIRST=compiletorusy ;;
    *) echo "usage"; exit 1 ;; esac
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $VAR $BK || { echo STOPPED; exit 1; }
run pcsv_${TAG}_${FIRST}_13129088_trial fabric_exp/pcs3/$FIRST.yaml $C O3 || { echo STOPPED; exit 1; }
for r in "$@"; do
    lab=pcsv_${TAG}_${r}_13129088_trial
    rm -f $out/$lab.log $out/pcsv_${TAG}_${r}_samples.bin $out/pcsv_${TAG}_${r}_samples.json
    ( run $lab fabric_exp/pcs3/$r.yaml $C O3 ) & RP=$!
    timeout 300 python_env/bin/python $CON/pc_sampler.py $out/pcsv_${TAG}_${r}_samples --start-log $out/$lab.log \
        --max-s 120 2>&1 | grep --line-buffered -E '^sampler: [0-9]+ samples|Error|error|Traceback'
    wait $RP || { echo "!!! $r failed"; break; }
    awk -F, -v r=$r 'NR>1 {printf "%s: %s %s %.2f cycles/packet\n", r, $1, $3, 1.35e9/$10}' $out/$lab.csv
done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
