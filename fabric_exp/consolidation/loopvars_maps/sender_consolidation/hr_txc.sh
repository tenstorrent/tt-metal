#!/bin/bash
# hr_txc.sh BASE (sender consolidation; sender_pass/hr_tx.sh writing here): the sender headroom check
# (V4_SENDER_OWNERSHIP.md §7.4): BASE's synthetic overlays (synth_overlays_tx.py:
# BASE_tx1, _tx2, _twk, _trr, made beforehand) built at -O3 on the families, then loopvars.py --headroom --role sender
# against BASE's own full build. Output: loopvars_maps/sender_consolidation/headroom_BASE.txt and hr_BASE_<kind>.json.
set -u
B=$1
D=/home/aagarwal/tt-metal/fabric_exp/consolidation/loopvars_maps/sender_consolidation
FAM=bh_1d,bh_ring,bh_2d,bh_torus,bh_2d_vc2,bh_torus_vc2,bh_vc1,glx_express,bh_2d_vc2_trim_w0,bh_vc1_trim_w0
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad
cd /home/aagarwal/tt-metal
for k in tx1 tx2 twk trr; do
    python3 $S/rxo/swapguard.py fabric_exp/variants/${B}_$k || { echo "swap hazard"; exit 1; }
    env TT_COV_DIR=/home/aagarwal/.tt-metal-v3r-builds/cov_m OFFLINE_BASE=/home/aagarwal/.tt-metal-p1-base-m nice -n 10 \
        python3 fabric_exp/consolidation/coverage_txo.py build --variant fabric_exp/variants/${B}_$k --opt O3 \
        --name ${B}_$k --cases $FAM > $D/hr_build_${B}_$k.log 2>&1
    echo "${B}_$k: $(tail -n 1 $D/hr_build_${B}_$k.log)"
done
: > $D/headroom_$B.txt
for k in tx1 tx2 twk trr; do
    echo "== $B + $k" >> $D/headroom_$B.txt
    python3 fabric_exp/consolidation/loopvars.py --headroom /home/aagarwal/.tt-metal-v3r-builds/cov_m/builds/${B}_O3 \
        /home/aagarwal/.tt-metal-v3r-builds/cov_m/builds/${B}_${k}_O3 --role sender --cases $FAM \
        --json $D/hr_${B}_$k.json >> $D/headroom_$B.txt 2>&1
done
cat $D/headroom_$B.txt
