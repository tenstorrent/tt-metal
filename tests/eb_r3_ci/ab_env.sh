#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary: a test selection with an environment toggle set ("main") and unset ("optin"), runs main optin
# optin main under the device profiler, plus one run of each side with the bits plugin. usage: ab_env.sh <VAR> <pytest args...>
set -uo pipefail
cd /work
VAR=$1; shift
ARGS=(); while (( $# )); do if [[ "$1" == "-k" ]]; then export EB_K_EXPR="$2"; ARGS+=(-p eb_k_plugin); shift 2; else ARGS+=("$1"); shift; fi; done; set -- "${ARGS[@]}"  # tracy splits a -k expression at spaces
O=/tmp/ebenv; mkdir -p $O
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
run=0
for v in main optin optin main; do
  run=$((run+1))
  if [[ $v == main ]]; then export $VAR=1; else unset $VAR; fi
  export TT_METAL_CACHE=$O/cache_$v TT_METAL_PROFILER_DIR=$O/profraw_$run
  mkdir -p "$TT_METAL_CACHE" "$TT_METAL_PROFILER_DIR"
  OUT=$O/out_env_${run}_$v; rm -rf "$OUT"
  timeout -s INT -k 60 ${EB_RUN_LIMIT:-2400} python3 -m tracy -r -p --no-web-server -o "$OUT" -m pytest -p eb_prof_plugin -p no:cacheprovider -o timeout_method=thread -q -rfEs "$@" > $O/log_${run}_$v.txt 2>&1
  echo "--- run $run $v rc=$?: $(grep -E 'passed|failed|skipped|error' $O/log_${run}_$v.txt | tail -1)"
  grep -E "^(FAILED|ERROR)" $O/log_${run}_$v.txt | cut -c1-200 | head -10
done
python3 /work/tests/eb_r3_ci/prof_reduce.py $O env 2>&1
for v in main optin; do
  if [[ $v == main ]]; then export $VAR=1; else unset $VAR; fi
  export TT_METAL_CACHE=$O/cache_$v EB_HASH_OUT=$O/hash_$v.json
  timeout -s INT -k 60 ${EB_RUN_LIMIT:-2400} python3 -m pytest -p eb_bits_plugin -p no:cacheprovider -o timeout_method=thread -q -rfE "$@" > $O/bits_$v.txt 2>&1
  echo "--- bits $v rc=$?: $(grep -E 'passed|failed|error' $O/bits_$v.txt | tail -1)"
done
python3 - <<'PY'
import json
a = json.load(open("/tmp/ebenv/hash_main.json")); b = json.load(open("/tmp/ebenv/hash_optin.json"))
tests = sorted(set(a["outcome"]) | set(b["outcome"]))
for t in tests:
    ha, hb = a["hashes"].get(t), b["hashes"].get(t)
    print(("IDENTICAL " if ha == hb else "DIFFER    ") + t.split("::")[-1] + f"  outcome {a['outcome'].get(t)}/{b['outcome'].get(t)}")
PY
echo "=== done"
