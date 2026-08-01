#!/bin/bash
# Watches a run_loop*.sh's current iteration log for staleness WITHOUT ever
# touching the test process while it might just be slow (kernel compile,
# device bring-up, etc). Only once a stall is confirmed do we: run
# tt-triage.py -vv, wait for that log to be fully written, THEN kill the
# hung test and reset the device. The hung process is never touched before
# triage has been captured.
#
# Run this in a second terminal/tmux pane alongside run_loop.sh (or
# run_loop_reset_each.sh) -- it's plain bash, no special tooling required.
#
# Usage: ./poller_loop.sh [DONE_MARKER]
#   DONE_MARKER defaults to "NO_HANG_AFTER_", which both run_loop.sh and
#   run_loop_reset_each.sh's completion lines contain. If you run BOTH
#   loops back-to-back against the same LOGDIR, pass a more specific marker
#   for the second poller invocation (e.g. a substring only the second
#   loop's completion line contains) -- otherwise it will match the first
#   loop's leftover completion line in summary.log and report done
#   immediately without ever having watched the second loop. (Learned this
#   the hard way -- see FINDINGS.md.)
# Env:   GLM_HANG_LOGDIR to override the default log directory (must match
#   what the runner script used).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

LOGDIR="${GLM_HANG_LOGDIR:-$REPO_ROOT/generated/glm_chunked_hang_repro}"
CURRENT="$LOGDIR/current_run.info"
SUMMARY="$LOGDIR/summary.log"
CHECK_INTERVAL=20
STALL_THRESHOLD=300   # 5 minutes of no log growth while alive => hang
DONE_MARKER="${1:-NO_HANG_AFTER_}"

last_pid=""
last_size=-1
stalled_for=0

while true; do
  if grep -q "$DONE_MARKER" "$SUMMARY" 2>/dev/null; then
    echo "NO_HANG_FOUND after full loop, no stall ever seen"
    exit 0
  fi

  if [ ! -f "$CURRENT" ]; then
    sleep "$CHECK_INTERVAL"
    continue
  fi

  line=$(tail -1 "$CURRENT")
  pid=$(echo "$line" | grep -oP 'pid=\K[0-9]+' || true)
  log=$(echo "$line" | grep -oP 'log=\K\S+' || true)
  is_finished_line=$(echo "$line" | grep -c 'finished=' || true)

  if [ -z "$pid" ] || [ "$is_finished_line" -ge 1 ]; then
    sleep "$CHECK_INTERVAL"
    continue
  fi

  if [ "$pid" != "$last_pid" ]; then
    last_pid="$pid"
    last_size=-1
    stalled_for=0
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    sleep "$CHECK_INTERVAL"
    continue
  fi

  cur_size=$(stat -c%s "$log" 2>/dev/null || echo 0)
  if [ "$cur_size" = "$last_size" ]; then
    stalled_for=$((stalled_for + CHECK_INTERVAL))
  else
    stalled_for=0
  fi
  last_size=$cur_size

  if [ "$stalled_for" -ge "$STALL_THRESHOLD" ]; then
    echo "HANG_DETECTED pid=$pid log=$log stalled_for=${stalled_for}s -- running triage before touching it"

    ts=$(date -u +%Y%m%dT%H%M%SZ)
    TRIAGE_LOG="$LOGDIR/triage_pid${pid}_${ts}.log"
    (
      cd "$REPO_ROOT"
      if [ -f "$REPO_ROOT/python_env/bin/activate" ]; then
        source "$REPO_ROOT/python_env/bin/activate"
      fi
      ./tools/tt-triage.py -vv
    ) > "$TRIAGE_LOG" 2>&1
    triage_rc=$?

    if [ ! -s "$TRIAGE_LOG" ]; then
      sleep 3
    fi
    echo "TRIAGE_SAVED $TRIAGE_LOG rc=$triage_rc size=$(stat -c%s "$TRIAGE_LOG" 2>/dev/null || echo 0)"

    echo "killing hung test (pid=$pid) now that triage is captured"
    pkill -9 -f "test_glm_prefill_transformer_chunked_no_pcc" 2>/dev/null
    kill -9 "$pid" 2>/dev/null
    if [ -f "$LOGDIR/runner.pid" ]; then
      kill -9 "$(cat "$LOGDIR/runner.pid")" 2>/dev/null
    fi

    RESET_LOG="$LOGDIR/glx_reset_after_hang_${ts}.log"
    echo "resetting device: tt-smi -glx_reset"
    tt-smi -glx_reset > "$RESET_LOG" 2>&1
    reset_rc=$?

    echo "HANG_HANDLED pid=$pid log=$log triage=$TRIAGE_LOG reset_log=$RESET_LOG reset_rc=$reset_rc"
    printf 'pid=%s\nlog=%s\ntriage=%s\nreset_log=%s\nreset_rc=%s\n' "$pid" "$log" "$TRIAGE_LOG" "$RESET_LOG" "$reset_rc" > "$LOGDIR/hang_detected.marker"
    exit 0
  fi

  sleep "$CHECK_INTERVAL"
done
