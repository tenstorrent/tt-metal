#!/usr/bin/env bash
# Usage: run_guarded.sh <name> <inactivity_seconds> <hard_timeout_seconds> -- <command...>
# Runs a device command with (a) a hard timeout and (b) an inactivity watchdog: if the log file
# stops growing for <inactivity_seconds> the process tree is killed (SIGKILL after SIGTERM) and
# the run is reported as HUNG so the same configuration is not retried blindly. Logs go to
# results/guard/<name>.log; exit codes: 0 ok, 124 hard timeout, 125 hung (inactivity), else cmd's.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="$1"; IDLE="$2"; HARD="$3"; shift 3; [ "$1" = "--" ] && shift
mkdir -p "$HERE/results/guard"; LOG="$HERE/results/guard/$NAME.log"; : > "$LOG"
ulimit -u 65536 2>/dev/null || true
setsid bash -c "$*" > "$LOG" 2>&1 &
PID=$!
START=$(date +%s); LAST=$START; SIZE=0; RC=0
while kill -0 $PID 2>/dev/null; do
  sleep 5
  NOW=$(date +%s); NEWSIZE=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  if [ "$NEWSIZE" != "$SIZE" ]; then SIZE=$NEWSIZE; LAST=$NOW; fi
  if [ $((NOW - LAST)) -ge "$IDLE" ]; then echo "[guard] HUNG: no output for ${IDLE}s -> killing $NAME" | tee -a "$LOG"; RC=125; break; fi
  if [ $((NOW - START)) -ge "$HARD" ]; then echo "[guard] hard timeout ${HARD}s -> killing $NAME" | tee -a "$LOG"; RC=124; break; fi
done
if [ $RC -ne 0 ]; then kill -TERM -- -$PID 2>/dev/null; sleep 10; kill -KILL -- -$PID 2>/dev/null; sleep 2; echo "$NAME $RC $(date -Is)" >> "$HERE/results/guard/HUNG_RUNS.txt"; else wait $PID; RC=$?; fi
echo "[guard] $NAME finished rc=$RC after $(( $(date +%s) - START ))s"
exit $RC
