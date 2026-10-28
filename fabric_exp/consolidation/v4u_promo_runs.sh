#!/bin/bash
# v4u before it becomes the lead: the stream suite (data-checked), bursty senders (stream idle segments) and the
# flushed-signal mixes, with v4l as the control on the benchmark rows.
source /home/aagarwal/tt-metal/fabric_exp/consolidation/jobs_lib.sh
source /home/aagarwal/tt-metal/fabric_exp/consolidation/cap_lib.sh
V=fabric_exp/variants
for v in v4u v4l; do
    C=/home/aagarwal/.tt-metal-cache-$v
    [ $v = v4l ] && { C=/home/aagarwal/.tt-metal-cache-v4lc; }
    mkdir -p $C
    python3 $CON/swap.py apply $V/$v $BK || { echo STOPPED; exit 1; }
    l=$v; [ $v = v4l ] && l=v4lc
    [ $v = v4u ] && { run_cap ${l}_13129088_stream $(suite_yaml stream) $C 300 || { echo STOPPED; exit 1; }; }
    run_cap ${l}_13129088_burst $(suite_yaml burst) $C 300
    run_cap ${l}_13129088_flushmix $(suite_yaml flushmix) $C 300
    run_cap ${l}_13129088_flushmix4 $(suite_yaml flushmix4) $C 300
    python3 $CON/swap.py restore $BK || { echo STOPPED; exit 1; }
done
rm -f $M
