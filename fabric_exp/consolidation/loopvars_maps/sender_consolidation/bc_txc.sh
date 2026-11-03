#!/bin/bash
# bc_txc.sh NAME "O3 Os" [REF] (sender consolidation; sender_pass/bc_tx.sh writing here): full coverage build of
# fabric_exp/variants/NAME (coverage_txo.py, every case, non-fabric
# files guarded by swapguard.py) at each level, in parallel; then the ok-set check and bytecmp.py against REF's build
# (default v4rxg3). Logs and compare tables in loopvars_maps/sender_consolidation/.
set -u
N=$1; LEVELS=$2; REF=${3:-v4rxg3}
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad
D=/home/aagarwal/tt-metal/fabric_exp/consolidation/loopvars_maps/sender_consolidation
cd /home/aagarwal/tt-metal
python3 $S/rxo/swapguard.py fabric_exp/variants/$N || { echo "refusing to build: see above"; exit 1; }
for o in $LEVELS; do
    env TT_COV_DIR=/home/aagarwal/.tt-metal-v3r-builds/cov_m OFFLINE_BASE=/home/aagarwal/.tt-metal-p1-base-m nice -n 10 \
        python3 fabric_exp/consolidation/coverage_txo.py build --variant fabric_exp/variants/$N --opt $o --name $N \
        > $D/build_${N}_$o.log 2>&1 &
done
wait
python3 $S/rxo/swapguard.py fabric_exp/variants/$N || echo "!!! a swap became active during the build: rebuild before trusting it"
for o in $LEVELS; do
    echo "$N $o: $(tail -n 1 $D/build_${N}_$o.log)"
    python3 - "${REF}_$o" "${N}_$o" <<'PY'
import json, sys
a = json.load(open(f"/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds/{sys.argv[1]}/results.json"))
b = json.load(open(f"/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds/{sys.argv[2]}/results.json"))
bad = [k for k in b if k in a and b[k]["ok"] != a[k]["ok"]]
print(f"  configs {len(b)}, ok {sum(v['ok'] for v in b.values())}, not in base {len([k for k in b if k not in a])}")
print(f"  ok-set mismatches: {len(bad)}", bad[:3], [b[k].get('err', '')[:200] for k in bad[:2]])
PY
    python3 $S/rxbeauty/bytecmp.py ${REF}_$o ${N}_$o > $D/cmp_${N}_$o.txt; echo "  bytecmp exit $?"
    grep -v " differ    0" $D/cmp_${N}_$o.txt | head -40 | sed 's/^/  /'
done
