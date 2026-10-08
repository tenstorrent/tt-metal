#!/usr/bin/env bash
# Round 3 eltwise binary: device time A/B for tests that create their own submesh (the repetition plugin's extra calls abort
# at submesh close): each test id from a nodes file alone in its process under tracy, <passes> rounds of main optin optin
# main, every run its own kernel cache and profiler dir; prof_sum.py sums the op's launches per run.
# usage: ab_plain.sh <optin file> <spec file: "test file|-k expression" per line, one test each> <passes> <op code regex>
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
set -uo pipefail
cd /work
OPT=$1; NODES=$(readlink -f "$2"); PASSES=$3; OPRE=$4
O=/tmp/ebplain; rm -rf $O; mkdir -p $O
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
t=0
while read -r NODE; do
  [[ -z "$NODE" ]] && continue
  t=$((t+1)); TFILE=${NODE%%|*}; export EB_K_EXPR=${NODE#*|}; echo "--- test $t: $TFILE -k $EB_K_EXPR"
  run=0
  for p in $(seq 1 $PASSES); do
    for v in main optin optin main; do
      run=$((run+1)); restore; [[ $v == optin ]] && apply
      export TT_METAL_CACHE=$O/cache_$v TT_METAL_PROFILER_DIR=$O/profraw_${t}_$run
      mkdir -p "$TT_METAL_CACHE" "$TT_METAL_PROFILER_DIR"
      OUT=$O/out_${t}_${run}_$v
      timeout -s INT -k 60 ${EB_RUN_LIMIT:-1200} python3 -m tracy -r -p --no-web-server -o "$OUT" -m pytest -p eb_k_plugin -p no:cacheprovider -o timeout_method=thread -q -rfEs "$TFILE" < /dev/null > $O/log_${t}_${run}_$v.txt 2>&1
      echo "--- test $t run $run $v rc=$?: $(grep -E 'passed|failed|skipped|error' $O/log_${t}_${run}_$v.txt | tail -1)"
      grep -E "error:|TT_THROW|TT_FATAL|Aborted" $O/log_${t}_${run}_$v.txt | sort | uniq -c | sort -rn | head -4 | cut -c1-300
    done
  done
done < "$NODES"
restore
python3 /work/tests/eb_r3_ci/prof_sum.py $O "$OPRE"
echo "=== done"
