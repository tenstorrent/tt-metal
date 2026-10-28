#!/bin/bash
# Completion-credit counters on v4gi's burst losses (RESULTS §98: N/S bursts +4 to +6 vs v4vc, cause open). v4gidg and
# v4vcdg are v4gi and v4vc plus router_diag.hpp's counters, which compile away unless TT_FABRIC_TXQ_COUNTERS=1 has the
# builder allocate the phase-profile buffer (offline: loops identical to v4gi / v4vc without it). Each variant in a
# fresh kernel cache:
#   1. the long N/S 8 x 16 B burst row without the counters (the same-job rate control, labels dg_<v>_burstns8_off);
#   2. with the counters: the long rows burstns8, burstew8 (E/W control, no loss), burstns32, then the burst suite.
# Counters in results/dg_<v>_<row>_phase.csv, decoded by dg_report.py at the end.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
LABELS=""
for v in v4gidg v4vcdg; do
    C=/home/aagarwal/.tt-metal-cache-dg-$v
    [ -e $C ] && { echo "!!! $C exists"; exit 1; }
    mkdir -p $C
    python3 $CON/swap.py apply fabric_exp/variants/$v $BK || { echo STOPPED; exit 1; }
    EXTRA_ENV=
    run dg_${v}_burstns8_off fabric_exp/pcs3/burstns8.yaml $C O3 || { echo STOPPED; exit 1; }
    EXTRA_ENV=TT_FABRIC_TXQ_COUNTERS=1
    for r in burstns8 burstew8 burstns32; do
        run dg_${v}_${r} fabric_exp/pcs3/$r.yaml $C O3 || { echo STOPPED; exit 1; }
        LABELS="$LABELS dg_${v}_${r}"
    done
    run dg_${v}_burst $(suite_yaml burst) $C O3 || { echo STOPPED; exit 1; }
    LABELS="$LABELS dg_${v}_burst"
    EXTRA_ENV=
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
for v in v4gidg v4vcdg; do
    awk -F, -v l=dg_${v}_burstns8_off 'NR>1 {printf "%s: %s %.2f cycles/packet\n", l, $1, 1.35e9/$10}' \
        fabric_exp/results/dg_${v}_burstns8_off.csv
done
python3 $CON/dg_report.py $LABELS
rm -f $M
echo ALL_DONE
