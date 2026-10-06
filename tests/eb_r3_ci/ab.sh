#!/usr/bin/env bash
# Round 3 eltwise binary (#58724), CI on one Blackhole card: device time of a kernel as main has it ("main") against the
# same kernel with an opt-in define ("optin"), runs interleaved main optin optin main, each under the device profiler
# (tracy) with its own kernel cache and profiler directory. The kernel file is restored after every run.
# usage: ab.sh <spec file>; spec lines: <kernel path>|<define line>|<pytest args>
set -uo pipefail
cd /work
SPEC=$1; O=/tmp/ebci; mkdir -p $O
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
n=0
while IFS='|' read -r KFILE DEFINE ARGS; do
  [[ -z "$KFILE" || "$KFILE" == \#* ]] && continue
  n=$((n+1)); tag=s$n; cp "$KFILE" $O/orig_$n
  echo "=== spec $tag: $KFILE | $DEFINE | $ARGS"
  run=0
  for v in main optin optin main; do
    run=$((run+1)); cp $O/orig_$n "$KFILE"
    if [[ $v == optin ]]; then
      python3 - "$KFILE" "$DEFINE" <<'PY'
import sys
p, d = sys.argv[1], sys.argv[2]
s = open(p).read().splitlines(keepends=True)
i = next(k for k, l in enumerate(s) if l.startswith("#include"))
s.insert(i, d + "\n")
open(p, "w").write("".join(s))
PY
    fi
    export TT_METAL_CACHE=$O/cache_${tag}_$v TT_METAL_PROFILER_DIR=$O/profraw_${tag}_$run
    mkdir -p "$TT_METAL_CACHE" "$TT_METAL_PROFILER_DIR"
    OUT=$O/out_${tag}_${run}_$v; rm -rf "$OUT"
    timeout -s INT -k 60 ${EB_RUN_LIMIT:-1500} python3 -m tracy -r -p --no-web-server -o "$OUT" -m pytest -p eb_prof_plugin -p no:cacheprovider -o timeout_method=thread -q -rfEs $ARGS > $O/log_${tag}_${run}_$v.txt 2>&1
    echo "--- $tag run $run $v rc=$?: $(grep -E 'passed|failed|skipped|error' $O/log_${tag}_${run}_$v.txt | tail -1)"
    grep -E "^(FAILED|ERROR|SKIPPED)" $O/log_${tag}_${run}_$v.txt | head -6
  done
  cp $O/orig_$n "$KFILE"
  python3 /work/tests/eb_r3_ci/prof_reduce.py $O $tag 2>&1 | sed "s/^/[$tag] /"
done < "$SPEC"
echo "=== done"
