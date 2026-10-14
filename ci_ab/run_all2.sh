#!/usr/bin/env bash
# Scratch op-level A/B (17_CI_AB.md): one machine, variants' device files (ci_ab/var/<v>) swapped into the tree and the
# installed runtime root before each run, a fresh kernel cache per variant, every case in one tracy process per run,
# rotated rounds (ORDERS in ci_ab/job2.sh). Prints OPL <variant> <run> <case> <op> <ns> per measured call.
set +e +o pipefail
R=$(git rev-parse --show-toplevel); S=$R/ci_ab; cd $R
python3 -c "import tracy" 2>/dev/null || export PYTHONPATH=$R:$R/tools
export TT_METAL_HOME=$R TRACY_NO_WEB_SERVER=1 HWLOCK_HELD=ci R3_NMEAS=${R3_NMEAS:-6}
RT=$(cd /tmp && env -u PYTHONPATH python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))" 2>/dev/null)
source $S/job2.sh
FILES=$(for v in $(ls $S/var); do (cd $S/var/$v && find . -type f | sed 's#^\./##'); done | sort -u)
echo "== runner $(hostname) $(date -u +%FT%TZ) head $(git log -1 --format=%h) runtime root $RT; $(echo $FILES | wc -w) files"
mkdir -p /tmp/base; for f in $FILES; do mkdir -p /tmp/base/$(dirname $f); git show HEAD:$f > /tmp/base/$f 2>/dev/null; done
apply() { local f src; for f in $FILES; do src=/tmp/base/$f; [[ -f $S/var/$1/$f ]] && src=$S/var/$1/$f; cp $src $R/$f; [[ -f $RT/$f ]] && cp $src $RT/$f; done; }
rm -rf /tmp/ttc_* /tmp/olp; mkdir -p /tmp/olp; n=0
for order in "${ORDERS[@]}"; do
  for v in $order; do
    n=$((n+1)); vv=${v%2}; apply $vv; out=/tmp/olp/${v}_$n; t0=$(date +%s)
    TT_METAL_CACHE=/tmp/ttc_$v timeout 2400 python3 -m tracy -r -p --no-web-server -o $out $S/tools/prof_all.py $CASES > $out.log 2>&1
    echo "== run $n $v rc=$? $(( $(date +%s) - t0 )) s: $(grep -c 'CASE .* ok' $out.log) ok, $(grep -c 'CASE .* FAILED' $out.log) failed"
    grep 'CASE .* FAILED' $out.log | head -3
    python3 - "$out" "$v" "$n" <<'PY'
import csv, glob, sys
out, v, n = sys.argv[1:4]
fs = glob.glob(out + "/reports/*/ops_perf_results_*.csv")
if not fs:
    print(f"OPL {v} {n} NO_REPORT"); sys.exit()
X = ["DEVICE NCRISC KERNEL DURATION [ns]", "DEVICE TRISC1 KERNEL DURATION [ns]", "DEVICE BRISC KERNEL DURATION [ns]"]
case = None
for r in csv.DictReader(open(fs[0])):
    if r.get("OP TYPE") == "signpost":
        case = r["OP CODE"]; continue
    d = r.get("DEVICE KERNEL DURATION [ns]")
    if case and d:
        print(f"OPL {v} {n} {case} {r['OP CODE'].replace(' ', '_')} {d}")
        print(f"OPX {v} {n} {case} {r['OP CODE'].replace(' ', '_')} " + " ".join(r.get(c) or "na" for c in X))
PY
  done
done
apply ${ORDERS[0]%% *}
echo "== done $(date -u +%T)"
