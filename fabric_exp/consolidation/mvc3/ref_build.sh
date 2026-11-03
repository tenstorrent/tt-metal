#!/bin/bash
# mvc3 offline check, part 2: v4rxg3 rebuilt through coverage_txo.py (the builder v4rxg3v was built with) as v4rxg3txo,
# so the identity check compares like with like; then bytecmp v4rxg3txo -> v4rxg3v and v4rxg3 (its stored build) ->
# v4rxg3txo (the builder's own effect).
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad
D=/home/aagarwal/tt-metal/fabric_exp/consolidation/mvc3
cd /home/aagarwal/tt-metal
python3 $S/rxo/swapguard.py fabric_exp/variants/v4rxg3 || exit 1
for o in O3 Os; do
    env TT_COV_DIR=/home/aagarwal/.tt-metal-v3r-builds/cov_m OFFLINE_BASE=/home/aagarwal/.tt-metal-p1-base-m nice -n 10 \
        python3 fabric_exp/consolidation/coverage_txo.py build --variant fabric_exp/variants/v4rxg3 --opt $o --name v4rxg3txo \
        > $D/build_v4rxg3txo_$o.log 2>&1 &
done
wait
python3 $S/rxo/swapguard.py fabric_exp/variants/v4rxg3 || echo "!!! a swap became active during the build"
for o in O3 Os; do
    echo "== $o: $(tail -n 1 $D/build_v4rxg3txo_$o.log)"
    python3 $S/rxbeauty/bytecmp.py v4rxg3txo_$o v4rxg3v_$o > $D/cmp_v4rxg3txo_v4rxg3v_$o.txt; echo "  txo ref -> probe: exit $?"
    tail -n 3 $D/cmp_v4rxg3txo_v4rxg3v_$o.txt | head -1
    grep -v " differ    0" $D/cmp_v4rxg3txo_v4rxg3v_$o.txt | grep -v "^  differs\|^shared" | head
    python3 $S/rxbeauty/bytecmp.py v4rxg3_$o v4rxg3txo_$o > $D/cmp_v4rxg3_v4rxg3txo_$o.txt; echo "  stored ref -> txo ref: exit $?"
    grep "^shared" $D/cmp_v4rxg3_v4rxg3txo_$o.txt
    cmp <(grep "^  differs" $D/cmp_v4rxg3_v4rxg3txo_$o.txt) <(grep "^  differs" $D/cmp_v4rxg3_v4rxg3v_$o.txt) && echo "  same first differing ELFs as stored ref -> probe"
done
