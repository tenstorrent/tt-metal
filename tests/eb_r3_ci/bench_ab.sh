#!/usr/bin/env bash
# Round 3 eltwise binary: a test's own BenchmarkProfiler timings (eb_bench_plugin), main's kernels against an opt-in, without
# the profiler: <passes> rounds of main optin optin main, each a fresh process with its own kernel cache.
# usage: bench_ab.sh <optin file: "path|define line" per line> <passes> <pytest args...>
set -uo pipefail
cd /work
OPT=$1; PASSES=$2; shift 2
O=/tmp/ebbench; mkdir -p $O; rm -f $O/bench_*.txt
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
declare -a FILES
while IFS='|' read -r KFILE DEFINE; do [[ -z "$KFILE" || "$KFILE" == \#* ]] && continue; FILES+=("$KFILE|$DEFINE"); cp "$KFILE" "$O/$(echo $KFILE | tr / _).orig"; done < "$OPT"
apply() {
  for e in "${FILES[@]}"; do KFILE=${e%%|*}; DEFINE=${e#*|}
    python3 - "$KFILE" "$DEFINE" <<'PY'
import sys
p, d = sys.argv[1], sys.argv[2]
s = open(p).read().splitlines(keepends=True)
if not any(l.strip() == d for l in s):
    i = next(k for k, l in enumerate(s) if l.startswith("#include"))
    s.insert(i, d + "\n")
open(p, "w").write("".join(s))
PY
  done
}
restore() { for e in "${FILES[@]}"; do KFILE=${e%%|*}; cp "$O/$(echo $KFILE | tr / _).orig" "$KFILE"; done; }
run=0
for p in $(seq 1 $PASSES); do
  for v in main optin optin main; do
    run=$((run+1)); restore; [[ $v == optin ]] && apply
    export TT_METAL_CACHE=$O/cache_$v; mkdir -p "$TT_METAL_CACHE"
    timeout -s INT -k 60 ${EB_RUN_LIMIT:-900} python3 -m pytest -s -p eb_bench_plugin -p no:cacheprovider -o timeout_method=thread -q -rfEs "$@" > $O/log_${run}_$v.txt 2>&1
    echo "--- run $run $v rc=$?: $(grep -E 'passed|failed|skipped|error' $O/log_${run}_$v.txt | tail -1)"
    grep -E "^(FAILED|ERROR)" $O/log_${run}_$v.txt | cut -c1-200 | head -5
    grep -a "EB_BENCH" $O/log_${run}_$v.txt | sed "s/^.*EB_BENCH/EB_BENCH $v $run/" | tee -a $O/bench_all.txt
  done
done
restore
python3 - <<'PY'
import collections, statistics
d = collections.defaultdict(lambda: collections.defaultdict(list))
for l in open("/tmp/ebbench/bench_all.txt"):
    f = l.split()
    if len(f) < 6:
        continue
    v, test, step, us = f[1], f[3], f[4], float(f[5])
    d[(test.split("::")[-1], step)][v].append(us)
for (t, s), vv in d.items():
    m, o = vv.get("main", []), vv.get("optin", [])
    if not m or not o:
        continue
    mm, om = statistics.median(m), statistics.median(o)
    rng = max(max(m) - min(m), max(o) - min(o))
    verdict = "equal" if abs(om - mm) <= rng else ("faster" if om < mm else "slower")
    print(f"BENCH {t} {s}: main n={len(m)} median {mm:.1f} us [{min(m):.1f}..{max(m):.1f}]  optin n={len(o)} median {om:.1f} us [{min(o):.1f}..{max(o):.1f}]  change {om - mm:+.1f} us ({100 * (om - mm) / mm:+.2f} %) range {rng:.1f} {verdict}")
PY
echo "=== done"
