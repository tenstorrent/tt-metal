#!/usr/bin/env bash
# CONCURRENCY, MEASURED -- four schedules, real timestamps, one job.
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
SEL=(--splits 5 --group 1 .)

echo "===== compile  $(date -u +%H:%M:%S)"
PERF_RUN_TAG=compile pytest $PQ --compile-producer -n 10 -m "$M" --timeout=60 \
  "${SEL[@]}" > /tmp/c.log 2>&1 || echo "  (producer rc=$?)"
tail -2 /tmp/c.log | sed 's/^/  /'

pass() {
  local label="$1"; shift
  echo "===== $label  $(date -u +%H:%M:%S)  [$*]"
  rm -f /tmp/corelog.*
  PERF_RUN_TAG="$label" PERF_CORE_LOG=/tmp/corelog \
    pytest $PQ --compile-consumer -n 15 -m "$M" --timeout=60 "$@" "${SEL[@]}" \
    > "/tmp/$label.log" 2>&1 || echo "  (rc=$?)"
  tail -2 "/tmp/$label.log" | sed 's/^/  /'
  local dest="$LLK_ROOT/perf_data/runs/corelog-$label"
  mkdir -p "$dest"
  cat /tmp/corelog.*.tsv > "$dest/cores.tsv" 2>/dev/null || true
  echo "  core log lines: $(wc -l < "$dest/cores.tsv" 2>/dev/null || echo 0)"
}

# Reference: the schedule that fires.
pass c374_a --maxschedchunk 374
pass c374_b --maxschedchunk 374
# Counter-example 1: whole files.
pass loadfile_a --dist loadfile
pass loadfile_b --dist loadfile
# Counter-example 2: crc32 hash groups, one per worker.
PERF_STABLE_GROUPS=15 pass crc32_a --dist loadgroup
PERF_STABLE_GROUPS=15 pass crc32_b --dist loadgroup

echo "===== runs:"; ls -1 "$LLK_ROOT/perf_data/runs/"
echo "===== done ====="
