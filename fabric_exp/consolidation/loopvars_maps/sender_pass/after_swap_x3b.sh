#!/bin/bash
# Waits until no live swap touches files these variants do not pin, then: X3b's full build and compare against X3a, its
# guard at both levels, the -Os fence check, and the sender headroom of X3, X3k, X3a, X3b.
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad
D=/home/aagarwal/tt-metal/fabric_exp/consolidation/loopvars_maps/sender_pass
cd /home/aagarwal/tt-metal
until python3 $S/rxo/swapguard.py fabric_exp/variants/v4txo3b > /dev/null; do sleep 60; done
echo "swap clear at $(date +%H:%M)"
bash $D/bc_tx.sh v4txo3b "O3 Os" v4txo3a > $D/bc_v4txo3b.log 2>&1
grep -E "built|ok-set|shared|bytecmp|!!!" $D/bc_v4txo3b.log
for o in O3 Os; do timeout 3000 python3 fabric_exp/consolidation/loopvars_maps/structure/guard_builds.py v4txo3a_$o v4txo3b_$o > $D/guard_v4txo3a_v4txo3b_$o.txt 2>&1; done
until python3 $S/rxo/swapguard.py fabric_exp/variants/v4txo0f1 > /dev/null; do sleep 60; done
bash $D/bc_tx.sh v4txo0f1 "Os" v4txo0 > $D/bc_v4txo0f1_Os.log 2>&1
grep -E "built|ok-set|shared|bytecmp|!!!|HAZARD" $D/bc_v4txo0f1_Os.log
for v in v4txo3 v4txo3k v4txo3a v4txo3b; do
    until python3 $S/rxo/swapguard.py fabric_exp/variants/$v > /dev/null; do sleep 60; done
    rm -rf /home/aagarwal/.tt-metal-v3r-builds/cov_m/builds/${v}_t*_O3
    bash $D/hr_tx.sh $v > /dev/null 2>&1
    grep -c "pass" $D/headroom_$v.txt
done
echo DONE
