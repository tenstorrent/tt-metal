#!/bin/bash
# mvc3_offline.sh [NAME] [REF] [LEVELS]: offline checks of the VC1 + VC2 probe for the three-VC throughput rows
# (2026-11-02; report multi-VC page). Default NAME v4rxg3v (= v4rxg3 with `!has_vc1` dropped from vc2_config_supported),
# REF v4rxg3, LEVELS "O3 Os". Full coverage build of fabric_exp/variants/NAME through coverage_txo.py (every cov_m case,
# the builder bc_tx.sh uses), non-fabric files guarded by swapguard.py before and after; then
#   - the ok-set against REF's build (only the VC1 + VC2 cases may change: they fail on REF by its gate),
#   - bytecmp.py REF NAME: every image both builds have must be byte-identical,
#   - size_report.py at each level (REF first), VC1 + VC2 cases included.
# Logs and tables in fabric_exp/consolidation/mvc3/.
set -u
N=${1:-v4rxg3v}; REF=${2:-v4rxg3}; LEVELS=${3:-O3 Os}
S=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad
D=/home/aagarwal/tt-metal/fabric_exp/consolidation/mvc3
COVD=/home/aagarwal/.tt-metal-v3r-builds/cov_m
mkdir -p $D
cd /home/aagarwal/tt-metal
python3 $S/rxo/swapguard.py fabric_exp/variants/$N || { echo "refusing to build: see above"; exit 1; }
for o in $LEVELS; do
    env TT_COV_DIR=$COVD OFFLINE_BASE=/home/aagarwal/.tt-metal-p1-base-m nice -n 10 \
        python3 fabric_exp/consolidation/coverage_txo.py build --variant fabric_exp/variants/$N --opt $o --name $N \
        > $D/build_${N}_$o.log 2>&1 &
done
wait
python3 $S/rxo/swapguard.py fabric_exp/variants/$N || echo "!!! a swap became active during the build: rebuild before trusting it"
for o in $LEVELS; do
    echo "== $N -$o: $(tail -n 1 $D/build_${N}_$o.log)"
    python3 - "${REF}_$o" "${N}_$o" <<'PY'
import collections, json, sys
a = json.load(open(f"/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds/{sys.argv[1]}/results.json"))
b = json.load(open(f"/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds/{sys.argv[2]}/results.json"))
print(f"  configs {len(b)}, ok {sum(v['ok'] for v in b.values())} (ref ok {sum(v['ok'] for v in a.values())}), "
      f"not in ref {len([k for k in b if k not in a])}")
ch = collections.Counter((b[k]["case"], a[k]["ok"], b[k]["ok"]) for k in b if k in a and b[k]["ok"] != a[k]["ok"])
for (case, x, y), n in sorted(ch.items()):
    print(f"  ok changed: {case}: {x} -> {y}: {n}")
bad = collections.Counter(v["case"] for v in b.values() if not v["ok"])
print("  still failing:", dict(sorted(bad.items())))
errs = collections.Counter(v["err"] for v in b.values() if not v["ok"] and "vc1_vc2" in v["case"])
for e, n in errs.most_common(3):
    print(f"  VC1+VC2 failure x{n}: {e}")
PY
    python3 $S/rxbeauty/bytecmp.py ${REF}_$o ${N}_$o > $D/cmp_${N}_$o.txt; echo "  bytecmp exit $? (1 expected: the VC1+VC2 ELFs exist only in $N)"
    tail -n 25 $D/cmp_${N}_$o.txt | grep -v " differ    0" | sed 's/^/  /'
    TT_COV_DIR=$COVD python3 fabric_exp/consolidation/size_report.py --opt $o $REF $N > $D/size_${N}_$o.txt 2>&1
    echo "  size_report exit $? -> $D/size_${N}_$o.txt"
done
