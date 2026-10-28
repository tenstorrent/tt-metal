#!/bin/bash
# Bursts of 6 and 12 (exp_burst612.yaml): burst lengths that are not multiples of the completion group, so v4vc's
# batches (which restart at a gap) no longer end exactly at the next gap. v4gidg and v4vcdg (v4gi and v4vc, counters
# compiled away unless TT_FABRIC_TXQ_COUNTERS=1), each in a fresh kernel cache:
#   at the default worker channel (17 slots): burst612 and the burst suite (same-job reference), counters off;
#   with TT_FABRIC_EXP_WORKER_SLOTS=32: burst612, counters off (the window's share);
#   counters on, 17 slots: burst612.
# Labels dg3_<v>_<suite>_w17|w32|c17.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
LABELS=""
for v in v4gidg v4vcdg; do
    C=/home/aagarwal/.tt-metal-cache-dg3-$v
    [ -e $C ] && { echo "!!! $C exists"; exit 1; }
    mkdir -p $C
    python3 $CON/swap.py apply fabric_exp/variants/$v $BK || { echo STOPPED; exit 1; }
    EXTRA_ENV=
    run dg3_${v}_burst612_w17 $(suite_yaml burst612) $C O3 || { echo STOPPED; exit 1; }
    run dg3_${v}_burst_w17 $(suite_yaml burst) $C O3 || { echo STOPPED; exit 1; }
    EXTRA_ENV="TT_FABRIC_EXP_WORKER_SLOTS=32"
    run dg3_${v}_burst612_w32 $(suite_yaml burst612) $C O3 || { echo STOPPED; exit 1; }
    EXTRA_ENV="TT_FABRIC_TXQ_COUNTERS=1"
    run dg3_${v}_burst612_c17 $(suite_yaml burst612) $C O3 || { echo STOPPED; exit 1; }
    EXTRA_ENV=
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
    LABELS="$LABELS dg3_${v}_burst612_c17"
done
for s in burst612_w17 burst_w17 burst612_w32; do
    for v in v4gidg v4vcdg; do
        awk -F, -v l=dg3_${v}_$s 'NR>1 {printf "%s: %s %s B %.2f cycles/packet\n", l, $1, $7, 1.35e9/$10}' \
            fabric_exp/results/dg3_${v}_$s.csv
    done
done
python3 $CON/dg_report.py $LABELS
rm -f $M
echo ALL_DONE
