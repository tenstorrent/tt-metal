#!/bin/bash
# ack_main_pin.sh: main's router under the old and the new ACK_CMD_BUF_ID rule, built from mainx5pin (mainx5base, main's
# v1 router as shipped in the v2 slot with the live library, plus the base-state tt_metal/hw/inc files it includes, so a
# broker job's swap cannot reach the build; mainx5base read them from the live tree and its builds overlapped a swap).
# -O3 and -Os; byte compare; diff_src.py maps every differing image's changed instructions to source lines.
D=/home/aagarwal/tt-metal/fabric_exp/consolidation/loopvars_maps/sender_pass
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad
cd /home/aagarwal/tt-metal
E="TT_COV_DIR=/home/aagarwal/.tt-metal-v3r-builds/cov_m OFFLINE_BASE=/home/aagarwal/.tt-metal-p1-base-m"
for o in O3 Os; do
    python3 $S/rxo/swapguard.py fabric_exp/variants/mainx5pin || exit 1
    env $E nice -n 10 python3 fabric_exp/consolidation/coverage_txo.py build --variant fabric_exp/variants/mainx5pin --opt $o --name mpnew > $D/ack_mpnew_$o.log 2>&1 &
    env $E OFFLINE_OLD_ACK_CMD_BUF=1 nice -n 10 python3 fabric_exp/consolidation/coverage_txo.py build --variant fabric_exp/variants/mainx5pin --opt $o --name mpold > $D/ack_mpold_$o.log 2>&1 &
    wait
    echo "$o: new $(tail -n 1 $D/ack_mpnew_$o.log); old $(tail -n 1 $D/ack_mpold_$o.log)"
    python3 - mpold_$o mpnew_$o <<'PY'
import json, sys
B = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
a = json.load(open(f"{B}/{sys.argv[1]}/results.json")); b = json.load(open(f"{B}/{sys.argv[2]}/results.json"))
bad = [k for k in a if k in b and a[k]["ok"] != b[k]["ok"]]
print(f"  ok-set mismatches: {len(bad)}", bad[:3])
PY
    python3 $S/rxbeauty/bytecmp.py mpold_$o mpnew_$o > $D/ack_cmp_mp_$o.txt; grep shared $D/ack_cmp_mp_$o.txt
    python3 $D/diff_src.py mpold_$o mpnew_$o 0 > $D/ack_src_mp_$o.txt 2>&1; tail -n 1 $D/ack_src_mp_$o.txt
done
python3 $D/diff_src.py hrold_Os hrnew_Os 0 > $D/ack_src_hr_Os.txt 2>&1; echo "hr Os: $(tail -n 1 $D/ack_src_hr_Os.txt)"
echo ACK_MAIN_DONE
