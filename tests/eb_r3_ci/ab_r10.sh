#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, rules of 16:40 / 17:40: a Blackhole model's own tests, main's program ("main": the listed kernel
# defines removed and MAIN_ENV set) against the head ("optin"), runs main optin optin main in one tree (path-matched) under
# the device profiler, EB_REPS repetitions per test; per op configuration (prof_reduce_cfg.py).
# usage: ab_r10.sh <off list or -> "<MAIN_ENV or ->" <pytest args...>
set -uo pipefail
cd /work
OFF=$1; MAIN_ENV=$2; shift 2
O=/tmp/ebr10; rm -rf $O; mkdir -p $O
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-} EB_REPS=${EB_REPS:-2}
declare -a FILES
if [[ $OFF != - ]]; then while IFS='|' read -r KFILE DEFINE; do [[ -z "$KFILE" || "$KFILE" == \#* ]] && continue; FILES+=("$KFILE|$DEFINE"); cp "$KFILE" "$O/$(echo $KFILE | tr / _).orig"; done < "$OFF"; fi
remove() { for e in "${FILES[@]}"; do KFILE=${e%%|*}; DEFINE=${e#*|}; python3 - "$KFILE" "$DEFINE" <<'PY'
import sys
p, d = sys.argv[1], sys.argv[2]
s = open(p).read().splitlines(keepends=True)
open(p, "w").write("".join(l for l in s if l.strip() != d.strip()))
PY
done; }
restore() { for e in "${FILES[@]}"; do KFILE=${e%%|*}; cp "$O/$(echo $KFILE | tr / _).orig" "$KFILE"; done; }
run=0
for v in main optin optin main; do
  run=$((run+1)); restore
  ENVS=""; if [[ $v == main ]]; then remove; [[ $MAIN_ENV != - ]] && ENVS="$MAIN_ENV"; fi
  export TT_METAL_CACHE=$O/cache_$v TT_METAL_PROFILER_DIR=$O/profraw_$run; mkdir -p "$TT_METAL_CACHE" "$TT_METAL_PROFILER_DIR"
  OUT=$O/out_r10_${run}_$v
  env $ENVS timeout -s INT -k 60 ${EB_RUN_LIMIT:-2400} python3 -m tracy -r -p --no-web-server -o "$OUT" -m pytest -p eb_prof_plugin -p no:cacheprovider -o timeout_method=thread -q -rfEs "$@" > $O/log_${run}_$v.txt 2>&1
  echo "--- run $run $v rc=$?: $(grep -E 'passed|failed|skipped|error' $O/log_${run}_$v.txt | tail -1 | cut -c1-200)"
  grep -E "^(FAILED|ERROR)" $O/log_${run}_$v.txt | cut -c1-200 | head -5
done
restore
python3 /work/tests/eb_r3_ci/prof_reduce_cfg.py $O r10 2>&1
echo "=== done"
