#!/bin/bash
# ack_main_os.sh: after ackrule_check.sh, main's router (mainx5base: main's v1 router as shipped, in the v2 slot, live
# library) at -Os under both rules, then diff_src.py (every differing image's changed instructions mapped to source
# lines) on v4rxg3 -Os and main -O3/-Os.
D=/home/aagarwal/tt-metal/fabric_exp/consolidation/loopvars_maps/sender_pass
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad
cd /home/aagarwal/tt-metal
while pgrep -f "sender_pass/ackrule_check.sh" > /dev/null; do sleep 20; done
E="TT_COV_DIR=/home/aagarwal/.tt-metal-v3r-builds/cov_m OFFLINE_BASE=/home/aagarwal/.tt-metal-p1-base-m"
python3 $S/rxo/swapguard.py fabric_exp/variants/mainx5base || exit 1
env $E nice -n 10 python3 fabric_exp/consolidation/coverage_txo.py build --variant fabric_exp/variants/mainx5base --opt Os --name mainnew > $D/ack_mainnew_Os.log 2>&1 &
env $E OFFLINE_OLD_ACK_CMD_BUF=1 nice -n 10 python3 fabric_exp/consolidation/coverage_txo.py build --variant fabric_exp/variants/mainx5base --opt Os --name mainold > $D/ack_mainold_Os.log 2>&1 &
python3 $D/diff_src.py mainold_O3 mainnew_O3 0 > $D/ack_src_main_O3.txt 2>&1 &
python3 $D/diff_src.py hrold_Os hrnew_Os 0 > $D/ack_src_hr_Os.txt 2>&1 &
wait
python3 $S/rxbeauty/bytecmp.py mainold_Os mainnew_Os > $D/ack_cmp_mainnew_Os.txt; tail -n 1 $D/ack_cmp_mainnew_Os.txt
python3 $D/diff_src.py mainold_Os mainnew_Os 0 > $D/ack_src_main_Os.txt 2>&1
for f in ack_src_hr_Os ack_src_main_O3 ack_src_main_Os; do echo "== $f"; tail -n 1 $D/$f.txt; done
echo ACK_MAIN_DONE
