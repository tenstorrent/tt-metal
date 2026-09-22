#!/usr/bin/env bash
# CORE SENSITIVITY -- the same tests on each Tensix in turn, one worker.
set -euo pipefail
GROUP="${1:?}"; N_GROUPS="${2:?}"
if [ "$GROUP" != "1" ]; then echo "only group 1 runs"; exit 0; fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLK_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR/python_tests"
export PERF_KEEP_RUNS=0
unset PERF_RUN_TAG
M="perf and not accuracy"
PQ="-q --override-ini=log_cli=false"
SEL=(--splits 25 --group 1 .)

echo "===== compile  $(date -u +%H:%M:%S)"
PERF_RUN_TAG=compile pytest $PQ --compile-producer -n 10 -m "$M" --timeout=60 \
  "${SEL[@]}" > /tmp/c.log 2>&1 || echo "  (producer rc=$?)"
tail -2 /tmp/c.log | sed 's/^/  /'

pass() {
  local label="$1" core="$2"
  echo "===== $label  core=$core  $(date -u +%H:%M:%S)"
  PERF_RUN_TAG="$label" PERF_FORCE_CORE_INDEX="$core" \
    pytest $PQ --compile-consumer -n 1 -m "$M" --timeout=60 "${SEL[@]}" \
    > "/tmp/$label.log" 2>&1 || echo "  (rc=$?)"
  tail -2 "/tmp/$label.log" | sed 's/^/  /'
}

for c in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do pass "core$c" "$c"; done
pass core0_repeat 0
echo "===== runs:"; ls -1 "$LLK_ROOT/perf_data/runs/"
echo "===== done ====="
