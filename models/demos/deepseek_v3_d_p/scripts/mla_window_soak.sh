#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Soak for the dev10 GDDR bank-3 DRAM corruption. Each bulk resets the 32-chip mesh, replays
# ring_mla, then reads the GDDR EDC counters before the next reset clears them. With EDC_PROBE=1
# the test freezes the mesh at the offending op and the dispatch-timeout hook runs tt-triage
# against it; the counter dump is the endpoint either way.
#
# Usage: [LOOPS=n] [ITERS=n] [EDC_PROBE=0|1] mla_window_soak.sh

set -uo pipefail

SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$SCRIPTS/../../../.." && pwd)}"
OUT="${OUT:-/data/$USER/soaklogs/mla_window_$(date +%Y%m%d_%H%M%S)}"
LOOPS="${LOOPS:-136}"
WATCH_DEV="${WATCH_DEV:-10}"
WATCH_INST="${WATCH_INST:-3}"
NODE=models/demos/deepseek_v3_d_p/tests/test_mla_window_repro.py::test_mla_window_repro
TRIAGE_ARGS="--run=dump_op_mesh --run=dump_callstacks --run=check_binary_integrity --llm-output -vv"

mkdir -p "$OUT"
cd "$TT_METAL_HOME"
source python_env/bin/activate

TSV="$OUT/iterations.tsv"
printf 'iter\tts\treset_s\trun_s\texit\tedc\tevent\n' > "$TSV"
echo "mla window soak -> $OUT   ${ITERS:-30000} iters/bulk, $LOOPS bulks, watching dev $WATCH_DEV inst $WATCH_INST"

for i in $(seq 1 "$LOOPS"); do
  ts=$(date -Is)

  r0=$(date +%s)
  tt-smi -glx_reset_auto > "$OUT/reset_$i.log" 2>&1
  reset_s=$(( $(date +%s) - r0 ))

  p0=$(date +%s)
  mkdir -p "$OUT/inspector_$i"
  EDC_PROBE="${EDC_PROBE:-1}" \
    MLA_WINDOW_ITERS="${ITERS:-30000}" \
    TT_METAL_OPERATION_TIMEOUT_SECONDS=5 \
    TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="python tools/tt-triage.py $TRIAGE_ARGS > $OUT/triage_$i.log 2>&1" \
    TT_METAL_INSPECTOR=1 \
    TT_METAL_LOGS_PATH="$OUT/inspector_$i" \
    pytest -q "$NODE" > "$OUT/run_$i.log" 2>&1
  rc=$?
  run_s=$(( $(date +%s) - p0 ))

  # Read before the next bulk's reset zeroes the counters. MRISC zeroes them after training, so
  # any non-zero value happened during this bulk.
  python3 "$SCRIPTS/gddr_edc_counters.py" --all > "$OUT/edc_$i.log" 2>&1
  if ! grep -q 'instances checked' "$OUT/edc_$i.log"; then
    edc=READ_ERR
  elif ! grep -q '\*\*\* EDC \*\*\*' "$OUT/edc_$i.log"; then
    edc=CLEAN
  elif grep -q "^dev *$WATCH_DEV inst $WATCH_INST:.*\*\*\* EDC \*\*\*" "$OUT/edc_$i.log"; then
    edc=DIRTY
  else
    edc=OTHER
  fi

  event=""
  [ "$edc" != CLEAN ] && event="EDC_$edc"
  [ "$rc" -ne 0 ] && event="${event:+$event,}EXIT_$rc"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$i" "$ts" "$reset_s" "$run_s" "$rc" "$edc" "${event:-CLEAN}" >> "$TSV"
  echo "  bulk $i: edc=$edc event=${event:-CLEAN}  run=${run_s}s exit=$rc"

  pkill -9 -f pytest 2>/dev/null || true
  # A crashed process stays registered against the device after it dies and the next bulk then
  # cannot map sysmem, so wait for the driver to drop entries whose process is gone. Only after a
  # crash -- the driver leaks dead entries, so unguarded this spins its full timeout every bulk.
  if [ "$rc" -ne 0 ]; then
    for _ in $(seq 1 90); do
      pgrep -f "[p]ytest -q $NODE" >/dev/null && { sleep 2; continue; }
      stale=0
      for p in $(sort -u /proc/driver/tenstorrent/*/pids 2>/dev/null); do
        kill -0 "$p" 2>/dev/null || stale=1
      done
      [ "$stale" -eq 0 ] && break
      sleep 2
    done
  fi
  sleep 5
done

echo "done at $(date), $LOOPS bulks -> $TSV"
column -t "$TSV"
