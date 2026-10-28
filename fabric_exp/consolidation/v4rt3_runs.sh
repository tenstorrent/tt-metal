#!/bin/bash
# The worker credit loop's round-trip breakdown (COMPLETION_CREDITS.md §2): v4girt / v4vcrt = v4gi / v4vc plus
# router_rt.hpp's sampled timestamps (E0: send, completion seen, credit written; E1: head, retire, credit, frame),
# compiled away unless TT_FABRIC_TXQ_COUNTERS=1 has the builder allocate the phase-profile buffer (offline: loops
# identical to v4gi / v4vc without it). Each variant in a fresh kernel cache:
#   off (rate controls, same job): burst612, burst, long N/S 8 x 16 B;
#   on: the same, the long N/S 32 x 4352 B row and the long E/W 3-hop 4352 B row.
# Layout control: v4gi itself with TT_FABRIC_TXQ_COUNTERS=1 (the buffer shifts L1; no hooks), burst and N/S 8 x 16 B.
# Labels rt3_<v>_<row>[_off] (rt_*: job 106, hooks perturbed +24 to +47; rt2_*: job 107, E0 sampled every 64th
# packet, aliased to the burst lengths, and E1 timed its own bookkeeping); decoded
# by rt_report.py at the end.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
LABELS=""
for v in v4girt v4vcrt v4gi; do
    C=/home/aagarwal/.tt-metal-cache-rt3-$v
    [ -e $C ] && { echo "!!! $C exists"; exit 1; }
    mkdir -p $C
    python3 $CON/swap.py apply fabric_exp/variants/$v $BK || { echo STOPPED; exit 1; }
    if [ $v = v4gi ]; then
        EXTRA_ENV="TT_FABRIC_TXQ_COUNTERS=1"
        run rt3_v4gi_burst_lay $(suite_yaml burst) $C O3 || { echo STOPPED; exit 1; }
        run rt3_v4gi_ns8_lay fabric_exp/pcs3/burstns8.yaml $C O3 || { echo STOPPED; exit 1; }
    else
        EXTRA_ENV=
        run rt3_${v}_burst612_off $(suite_yaml burst612) $C O3 || { echo STOPPED; exit 1; }
        run rt3_${v}_burst_off $(suite_yaml burst) $C O3 || { echo STOPPED; exit 1; }
        run rt3_${v}_ns8_off fabric_exp/pcs3/burstns8.yaml $C O3 || { echo STOPPED; exit 1; }
        EXTRA_ENV="TT_FABRIC_TXQ_COUNTERS=1"
        run rt3_${v}_burst612 $(suite_yaml burst612) $C O3 || { echo STOPPED; exit 1; }
        run rt3_${v}_burst $(suite_yaml burst) $C O3 || { echo STOPPED; exit 1; }
        run rt3_${v}_ns8 fabric_exp/pcs3/burstns8.yaml $C O3 || { echo STOPPED; exit 1; }
        run rt3_${v}_ns32 fabric_exp/pcs3/burstns32.yaml $C O3 || { echo STOPPED; exit 1; }
        run rt3_${v}_hop3 fabric_exp/pcs3/hop3.yaml $C O3 || { echo STOPPED; exit 1; }
        LABELS="$LABELS rt3_${v}_burst612 rt3_${v}_burst rt3_${v}_ns8 rt3_${v}_ns32 rt3_${v}_hop3"
    fi
    EXTRA_ENV=
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
for l in rt3_v4girt_ns8_off rt3_v4vcrt_ns8_off rt3_v4gi_ns8_lay; do
    awk -F, -v l=$l 'NR>1 {printf "%s: %s %s B %.2f cycles/packet\n", l, $1, $7, 1.35e9/$10}' fabric_exp/results/$l.csv
done
python3 $CON/rt_report.py $LABELS
rm -f $M
echo ALL_DONE
