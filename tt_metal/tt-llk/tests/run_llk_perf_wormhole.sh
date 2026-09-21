#!/usr/bin/env bash
# CHUNK DOSE-RESPONSE -- one shard, one job, chunk size the only variable.
#
# xdist chunk = max(2, len(pending)//nodes//2), capped by --maxschedchunk. With
# shard 1's 11,240 items over 15 workers the uncapped value is 374, which is the
# configuration that fires. Smaller caps are the workarounds we already know.
set -euo pipefail
GROUP="${1:?}"
N_GROUPS="${2:?}"
if [ "$GROUP" != "1" ]; then
  echo "experiment: only group 1 runs; this group exits."
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLK_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR/python_tests"
export PERF_KEEP_RUNS=0
unset PERF_RUN_TAG

M="perf and not accuracy"
PQ="-q --override-ini=log_cli=false"
SEL=(--splits 5 --group 1 .)

echo "===== compiling shard 1 once  $(date -u +%H:%M:%S)"
PERF_RUN_TAG=compile pytest $PQ --compile-producer -n 10 -m "$M" --timeout=60 \
  "${SEL[@]}" > /tmp/compile.log 2>&1 || echo "  (producer rc=$?)"
tail -2 /tmp/compile.log | sed 's/^/  /'

arm() {
  local label="$1" chunk="$2"
  echo "===== ARM $label  maxschedchunk=$chunk  $(date -u +%H:%M:%S)"
  export PERF_RUN_TAG="$label"
  pytest $PQ --compile-consumer -n 15 -m "$M" --timeout=60 \
    --maxschedchunk "$chunk" "${SEL[@]}" > "/tmp/$label.log" 2>&1 \
    || echo "  (consumer rc=$?)"
  tail -2 "/tmp/$label.log" | sed 's/^/  /'
  unset PERF_RUN_TAG
}

for c in 374 10 50 100 200; do
  arm "c${c}_a" "$c"
  arm "c${c}_b" "$c"
done

echo "===== arms written:"
ls -1 "$LLK_ROOT/perf_data/runs/" || true
echo "===== experiment done ====="
