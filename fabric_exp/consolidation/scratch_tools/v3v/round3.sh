#!/bin/bash
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3v; V=/home/aagarwal/tt-metal/fabric_exp/variants
cd $S
./hot.sh $V/v3ve $S/hotb/ve > $S/hotb/ve.log 2>&1
./tabx.sh $V/v3ve ve > $S/TX_ve.txt 2>&1
( ./fullcheck.sh v3v2 pVv2 > $S/check_v3v2.txt 2>&1 ) &
( ./fullcheck.sh v3v3 pP > $S/check_v3v3.txt 2>&1 ) &
( ./fullcheck.sh v3ve ve > $S/check_v3ve.txt 2>&1 ) &
# comment-only reflow in v3v / v3vc / v3vd: hot configs rebuilt and compared with the full-check builds
( for d in v3v v3vc v3vd; do ./hot.sh $V/$d $S/hotb/re_$d > /dev/null 2>&1; for l in O3 O2 Os; do echo "$d $l: $(python3 $S/loadcmp.py $S/full/${d}K_$l $S/hotb/re_${d}_$l subordinate_active_erisc active_erisc 2>&1 | tr '\n' ' ')"; done; done > $S/check_reflow.txt 2>&1 ) &
wait
echo round3-done
