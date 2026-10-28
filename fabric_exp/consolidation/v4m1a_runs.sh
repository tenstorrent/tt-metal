#!/bin/bash
# v4 router bring-up, first family (two-ERISC Blackhole, 1D and 2D, VC0, strict atomics), job A: smoke, then the
# data-checked validation suites at -O3 (fresh cache). A smoke hang ends in 5 minutes, not 20.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
V=fabric_exp/variants
C=/home/aagarwal/.tt-metal-cache-v4-O3
rm -rf $C; mkdir -p $C
python3 $CON/swap.py apply $V/v4 $BK || { echo STOPPED; exit 1; }
for y in compile2d compile1d; do
    env -u TT_METAL_SLOW_DISPATCH_MODE TT_FABRIC_KERNEL_VERSION=2 timeout 300 fabric_exp/run_exp.sh v4smoke_$y fabric_exp/pcs3/$y.yaml \
        TT_FABRIC_NO_BW_TELEMETRY=1 TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT=1000 TT_FABRIC_EXP_FLAGS=13129088 TT_METAL_CACHE=$C \
        TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 TT_METAL_FABRIC_OPT_LEVEL=O3 | grep --line-buffered -v 'Running Test'
    r=$(grep -oE 'Tests: .*executed, [0-9]+ failed' fabric_exp/results/v4smoke_$y.log)
    echo "v4smoke_$y: ${r:-NO RESULT}"
    [ -f fabric_exp/results/v4smoke_$y.csv ] && awk -F, 'NR>1{printf "   %s %.1f cycles/packet\n", $1, 1.35e9/$10}' fabric_exp/results/v4smoke_$y.csv
    echo "$r" | grep -q ", 0 failed" || { echo STOPPED; exit 1; }
done
python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
build v4_O3 $V/v4 $C O3 v4m1 mesh stream val1d_m1
rm -f $M
echo ALL_DONE
