#!/usr/bin/env bash
# Usage: devrun.sh <name> <inactivity_seconds> <hard_timeout_seconds> -- <command...>
# One device command at a time (lock), outside the sandbox, ulimit raised. Watchdog: no log growth for
# <inactivity_seconds> or exceeding <hard_timeout_seconds> -> process tree killed, devices RESET (tt-smi -r all).
# Exit: 0 ok, 124 hard timeout, 125 hung, else the command's. Log: $SPFUSE/logs/<name>.log; hangs -> logs/HUNG_RUNS.txt.
# NOTE: `python -m tracy -r` captures its child's output -> give profile runs idle == hard budget.
set -uo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/env.sh"
NAME="$1"; IDLE="$2"; HARD="$3"; shift 3; [ "${1:-}" = "--" ] && shift
LOG="$SPFUSE/logs/$NAME.log"
exec 8>"$SPFUSE/locks/device.lock"
echo "[devrun] $NAME waiting for device lock..."; flock 8; echo "[devrun] $NAME lock acquired $(date +%T)"
: > "$LOG"; cd "$TT_METAL_HOME"
setsid bash -c "$*" > "$LOG" 2>&1 &
PID=$!
START=$(date +%s); LAST=$START; SIZE=0; RC=0
while kill -0 $PID 2>/dev/null; do
  sleep 3
  NOW=$(date +%s); NEWSIZE=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  if [ "$NEWSIZE" != "$SIZE" ]; then SIZE=$NEWSIZE; LAST=$NOW; fi
  if [ $((NOW - LAST)) -ge "$IDLE" ]; then echo "[devrun] HUNG: no output for ${IDLE}s -> killing $NAME" | tee -a "$LOG"; RC=125; break; fi
  if [ $((NOW - START)) -ge "$HARD" ]; then echo "[devrun] hard timeout ${HARD}s -> killing $NAME" | tee -a "$LOG"; RC=124; break; fi
done
if [ $RC -ne 0 ]; then
  kill -TERM -- -$PID 2>/dev/null; sleep 8; kill -KILL -- -$PID 2>/dev/null; sleep 2
  echo "$NAME $RC $(date -Is)" >> "$SPFUSE/logs/HUNG_RUNS.txt"
  echo "[devrun] resetting devices after hang/timeout..." | tee -a "$LOG"; tt-smi -r all >> "$LOG" 2>&1; echo "[devrun] reset rc=$?" | tee -a "$LOG"
else
  wait $PID; RC=$?
fi
echo "[devrun] $NAME finished rc=$RC after $(( $(date +%s) - START ))s, log: $LOG"
tail -n 40 "$LOG" | grep -E "passed|failed|error|Error|PASS|FAIL|Traceback|TT_FATAL|TT_THROW|us/op|RESULT" | tail -25
exit $RC
