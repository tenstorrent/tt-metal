#!/bin/bash
# The host's new SENDER_CH_<i>_ACK_CMD_BUF_ID rule (host_rebuild_20261102b) against the old one, offline: v4rxg3 on full
# coverage at -O3 and -Os, main's router (mainx5base: main + the live library) at -O3 on every case; byte compare, then
# the guard on v4rxg3's differing images.
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad
D=/home/aagarwal/tt-metal/fabric_exp/consolidation/loopvars_maps/sender_pass
cd /home/aagarwal/tt-metal
E="TT_COV_DIR=/home/aagarwal/.tt-metal-v3r-builds/cov_m OFFLINE_BASE=/home/aagarwal/.tt-metal-p1-base-m"
python3 $S/rxo/swapguard.py fabric_exp/variants/v4rxg3hr || exit 1
for o in O3 Os; do
    env $E nice -n 10 python3 fabric_exp/consolidation/coverage_txo.py build --variant fabric_exp/variants/v4rxg3hr --opt $o --name hrnew > $D/ack_hrnew_$o.log 2>&1 &
    env $E OFFLINE_OLD_ACK_CMD_BUF=1 nice -n 10 python3 fabric_exp/consolidation/coverage_txo.py build --variant fabric_exp/variants/v4rxg3hr --opt $o --name hrold > $D/ack_hrold_$o.log 2>&1 &
    wait
done
env $E nice -n 10 python3 fabric_exp/consolidation/coverage_txo.py build --variant fabric_exp/variants/mainx5base --opt O3 --name mainnew > $D/ack_mainnew.log 2>&1 &
env $E OFFLINE_OLD_ACK_CMD_BUF=1 nice -n 10 python3 fabric_exp/consolidation/coverage_txo.py build --variant fabric_exp/variants/mainx5base --opt O3 --name mainold > $D/ack_mainold.log 2>&1 &
wait
python3 $S/rxo/swapguard.py fabric_exp/variants/v4rxg3hr || echo "!!! swap became active"
for p in "hrold_O3 hrnew_O3" "hrold_Os hrnew_Os" "mainold_O3 mainnew_O3"; do set -- $p; echo "== $1 vs $2"; python3 $S/rxbeauty/bytecmp.py $1 $2 > $D/ack_cmp_$2.txt; tail -n 1 $D/ack_cmp_$2.txt; done
for o in O3 Os; do timeout 3000 python3 fabric_exp/consolidation/loopvars_maps/structure/guard_builds.py hrold_$o hrnew_$o > $D/ack_guard_$o.txt 2>&1; head -1 $D/ack_guard_$o.txt; grep -v "loop +0 sp S+0 L+0$" $D/ack_guard_$o.txt | sed -n '2,$p' | cut -c1-120 | head; done
echo DONE
