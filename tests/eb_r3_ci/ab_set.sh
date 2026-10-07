#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary: device time of a test selection with main's kernels ("main") against the same tests with a set of
# opt-in defines ("optin"), runs main optin optin main, each under the device profiler with its own kernel cache and profiler
# directory; every kernel file is restored at the end. usage: ab_set.sh <optin file: "path|define line" per line> <pytest args...>
set -uo pipefail
cd /work
OPT=$1; shift
ARGS=(); while (( $# )); do if [[ "$1" == "-k" ]]; then export EB_K_EXPR="$2"; ARGS+=(-p eb_k_plugin); shift 2; else ARGS+=("$1"); shift; fi; done; set -- "${ARGS[@]}"  # tracy splits a -k expression at spaces
SEL=()
if [[ "${1:-}" == "--nodes-file" ]]; then export EB_NODES_FILE=$(readlink -f "$2"); shift 2; set -- "$@" $(cut -d: -f1 "$EB_NODES_FILE" | sort -u); SEL=(-p eb_select_plugin); fi
O=/tmp/ebset; mkdir -p $O
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
for v in main optin optin main; do
  run=$((run+1)); restore; [[ $v == optin ]] && apply
  export TT_METAL_CACHE=$O/cache_$v TT_METAL_PROFILER_DIR=$O/profraw_$run
  mkdir -p "$TT_METAL_CACHE" "$TT_METAL_PROFILER_DIR"
  OUT=$O/out_set_${run}_$v; rm -rf "$OUT"
  timeout -s INT -k 60 ${EB_RUN_LIMIT:-2400} python3 -m tracy -r -p --no-web-server -o "$OUT" -m pytest -p eb_prof_plugin "${SEL[@]}" -p no:cacheprovider -o timeout_method=thread -q -rfEs "$@" > $O/log_${run}_$v.txt 2>&1
  echo "--- run $run $v rc=$?: $(grep -E 'passed|failed|skipped|error' $O/log_${run}_$v.txt | tail -1)"
  grep -E "^(FAILED|ERROR)" $O/log_${run}_$v.txt | cut -c1-200 | head -10
  grep -q -E 'passed|failed|skipped|error' $O/log_${run}_$v.txt || tail -15 $O/log_${run}_$v.txt | cut -c1-200
  [[ -n "${EB_SHOW_ERR:-}" ]] && grep -E "error:|TT_THROW|TT_FATAL|Timeout|timed out|^E  " $O/log_${run}_$v.txt | sort | uniq -c | sort -rn | head -12 | cut -c1-500
done
restore
python3 /work/tests/eb_r3_ci/prof_reduce.py $O set 2>&1
echo "=== done"
