#!/bin/bash
# PC sampling for the bound map's "why" and the TX-queue probe's post-send work (2026-11-01; V4_BOUNDS.md): pcsv_runs.sh
# (pc_sampler.py over the debug bus, no router code) with row yamls from fabric_exp/pcs3/<row>.yaml or, when there,
# fabric_exp/consolidation/bnd/<row>.yaml (EW 1-hop writes at other sizes).
#   bnd_pcs_runs.sh VARIANT TAG 2D|1D|Ring|TorusY ROW...      (EXTRA_ENV adds settings, e.g. VC2 on)
# A fresh kernel cache: the first (short) row compiles the routers and logs their compile commands (for -g twins).
# Labels bndp_<tag>_<row>_13129088_trial (rate), samples results/bndp_<tag>_<row>_samples.*. A failed row stops the job.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
VAR=$1; TAG=$2; TOPO=$3; shift 3
out=fabric_exp/results
C=/home/aagarwal/.tt-metal-cache-bndp-$TAG
case "$TOPO" in 2D) FIRST=compile2d ;; 1D) FIRST=compile1d ;; Ring) FIRST=compilering ;; TorusY) FIRST=compiletorusy ;;
    *) echo "usage: bnd_pcs_runs.sh VARIANT TAG 2D|1D|Ring|TorusY ROW..."; exit 1 ;; esac
rm -rf "${C:?}"; mkdir -p $C
python3 $CON/swap.py apply $VAR $BK || { echo STOPPED; exit 1; }
run bndp_${TAG}_${FIRST}_13129088_trial fabric_exp/pcs3/$FIRST.yaml $C O3 || { python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
for r in "$@"; do
    y=fabric_exp/pcs3/$r.yaml
    [ -f $CON/bnd/$r.yaml ] && y=$CON/bnd/$r.yaml
    lab=bndp_${TAG}_${r}_13129088_trial
    rm -f $out/$lab.log $out/bndp_${TAG}_${r}_samples.bin $out/bndp_${TAG}_${r}_samples.json
    ( run $lab $y $C O3 ) & RP=$!
    timeout 300 python_env/bin/python $CON/pc_sampler.py $out/bndp_${TAG}_${r}_samples --start-log $out/$lab.log \
        --max-s 120 2>&1 | grep --line-buffered -E '^sampler: [0-9]+ samples|Error|error|Traceback'
    wait $RP || { echo "!!! $r failed"; python3 $CON/swap.py restore $BK; echo STOPPED; exit 1; }
    awk -F, -v r=$r 'NR>1 {printf "%s: %s %s %.2f cycles/packet\n", r, $1, $3, 1.35e9/$10}' $out/$lab.csv
done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
rm -f $M
echo ALL_DONE
