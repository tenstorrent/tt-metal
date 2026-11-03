#!/bin/bash
# X4 offline: X3f rebuilt under the new SENDER_CH_<i>_ACK_CMD_BUF_ID injection (v4txo3fn, same source), X4 against it at
# both levels, the guard, and X4's headroom.
D=/home/aagarwal/tt-metal/fabric_exp/consolidation/loopvars_maps/sender_pass
cd /home/aagarwal/tt-metal
bash $D/bc_tx.sh v4txo3fn "O3 Os" v4txo3f > $D/bc_v4txo3fn.log 2>&1
echo "== v4txo3fn (new injection) vs v4txo3f"; grep -E "built|ok-set|shared|bytecmp|!!!|HAZARD" $D/bc_v4txo3fn.log
bash $D/bc_tx.sh v4txo4 "O3 Os" v4txo3fn > $D/bc_v4txo4.log 2>&1
echo "== v4txo4 vs v4txo3fn"; grep -E "built|ok-set|shared|bytecmp|!!!|HAZARD" $D/bc_v4txo4.log
for o in O3 Os; do timeout 3000 python3 fabric_exp/consolidation/loopvars_maps/structure/guard_builds.py v4txo3fn_$o v4txo4_$o > $D/guard_v4txo3fn_v4txo4_$o.txt 2>&1; head -1 $D/guard_v4txo3fn_v4txo4_$o.txt; done
bash $D/hr_tx.sh v4txo4 > /dev/null 2>&1; grep -c pass $D/headroom_v4txo4.txt
echo DONE
