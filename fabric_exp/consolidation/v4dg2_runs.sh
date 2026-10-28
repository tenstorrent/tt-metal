#!/bin/bash
# Confirms job 103's reading of v4gi's burst losses: aligned groups leave a partial group waiting at the stream's gaps,
# which holds it 4 idle passes while the worker waits on its credits (2D worker channel: 17 slots, its credit window).
# Each variant in a fresh kernel cache, counters off unless named:
#   v4gidg, v4vcdg: v4gi, v4vc;  v4gidg1: v4gidg with partial groups released on the first idle pass (diagnostic).
#   at the default worker channel (17 slots): long N/S bursts of 8 x 16 B and 32 x 4352 B, the burst suite;
#   with TT_FABRIC_EXP_WORKER_SLOTS=32: the same (prediction: the v4gi - v4vc gap closes);
#   counters: N/S 8 x 16 B at 32 worker slots (all three), and v4gidg1's at 17.
# Labels dg2_<v>_<row>_w17|w32[c].
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
LABELS=""
CLABELS=""
for v in v4gidg v4vcdg v4gidg1; do
    C=/home/aagarwal/.tt-metal-cache-dg2-$v
    [ -e $C ] && { echo "!!! $C exists"; exit 1; }
    mkdir -p $C
    python3 $CON/swap.py apply fabric_exp/variants/$v $BK || { echo STOPPED; exit 1; }
    for w in 17 32; do
        if [ $w = 17 ]; then EXTRA_ENV="TT_FABRIC_EXP_LOG_SLOTS=1"; else EXTRA_ENV="TT_FABRIC_EXP_WORKER_SLOTS=32 TT_FABRIC_EXP_LOG_SLOTS=1"; fi
        run dg2_${v}_ns8_w$w fabric_exp/pcs3/burstns8.yaml $C O3 || { echo STOPPED; exit 1; }
        run dg2_${v}_ns32_w$w fabric_exp/pcs3/burstns32.yaml $C O3 || { echo STOPPED; exit 1; }
        run dg2_${v}_burst_w$w $(suite_yaml burst) $C O3 || { echo STOPPED; exit 1; }
        LABELS="$LABELS dg2_${v}_ns8_w$w dg2_${v}_ns32_w$w dg2_${v}_burst_w$w"
    done
    EXTRA_ENV="TT_FABRIC_EXP_WORKER_SLOTS=32 TT_FABRIC_TXQ_COUNTERS=1"
    run dg2_${v}_ns8_w32c fabric_exp/pcs3/burstns8.yaml $C O3 || { echo STOPPED; exit 1; }
    CLABELS="$CLABELS dg2_${v}_ns8_w32c"
    if [ $v = v4gidg1 ]; then
        EXTRA_ENV="TT_FABRIC_TXQ_COUNTERS=1"
        run dg2_${v}_ns8_w17c fabric_exp/pcs3/burstns8.yaml $C O3 || { echo STOPPED; exit 1; }
        CLABELS="$CLABELS dg2_${v}_ns8_w17c"
    fi
    EXTRA_ENV=
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
grep -h -i -m3 "slots" fabric_exp/results/dg2_v4gidg_ns8_w17.log fabric_exp/results/dg2_v4gidg_ns8_w32.log | cut -c1-250
for l in $LABELS; do
    awk -F, -v l=$l 'NR>1 {printf "%s: %s %s B %.2f cycles/packet\n", l, $1, $7, 1.35e9/$10}' fabric_exp/results/$l.csv
done
python3 $CON/dg_report.py $CLABELS
rm -f $M
echo ALL_DONE
