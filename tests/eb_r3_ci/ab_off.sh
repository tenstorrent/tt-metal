#!/usr/bin/env bash
# Round 3 eltwise binary: a test selection with listed kernel defines removed ("main": main's program) against the tree as is
# ("optin"), runs main optin optin main under the device profiler. usage: ab_off.sh <file: "path|exact define line"> <pytest args...>
set -uo pipefail
cd /work
OPT=$1; shift
O=/tmp/eboff; mkdir -p $O
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
declare -a FILES
while IFS='|' read -r KFILE DEFINE; do [[ -z "$KFILE" || "$KFILE" == \#* ]] && continue; FILES+=("$KFILE|$DEFINE"); cp "$KFILE" "$O/$(echo $KFILE | tr / _).orig"; done < "$OPT"
remove() {
  for e in "${FILES[@]}"; do KFILE=${e%%|*}; DEFINE=${e#*|}
    python3 - "$KFILE" "$DEFINE" <<'PY'
import sys
p, d = sys.argv[1], sys.argv[2]
s = open(p).read().splitlines(keepends=True)
n = [l for l in s if l.strip() != d.strip()]
assert len(n) == len(s) - 1, (p, d)
open(p, "w").write("".join(n))
PY
  done
}
restore() { for e in "${FILES[@]}"; do KFILE=${e%%|*}; cp "$O/$(echo $KFILE | tr / _).orig" "$KFILE"; done; }
run=0
for v in main optin optin main; do
  run=$((run+1)); restore; [[ $v == main ]] && remove
  export TT_METAL_CACHE=$O/cache_$v TT_METAL_PROFILER_DIR=$O/profraw_$run
  mkdir -p "$TT_METAL_CACHE" "$TT_METAL_PROFILER_DIR"
  OUT=$O/out_off_${run}_$v; rm -rf "$OUT"
  timeout -s INT -k 60 ${EB_RUN_LIMIT:-2400} python3 -m tracy -r -p --no-web-server -o "$OUT" -m pytest -p eb_prof_plugin -p no:cacheprovider -o timeout_method=thread -q -rfEs "$@" > $O/log_${run}_$v.txt 2>&1
  echo "--- run $run $v rc=$?: $(grep -E 'passed|failed|skipped|error' $O/log_${run}_$v.txt | tail -1)"
  grep -E "^(FAILED|ERROR)" $O/log_${run}_$v.txt | cut -c1-200 | head -10
done
restore
python3 /work/tests/eb_r3_ci/prof_reduce.py $O off 2>&1
echo "=== done"
