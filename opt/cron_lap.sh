#!/usr/bin/env bash
# Unstoppable LTX-opt loop: OS cron relaunches a FRESH headless claude agent each fire,
# so the loop survives Claude-process / session / reboot death (state lives in git + opt/PROGRESS.md).
# The in-session Monitor/CronCreate heartbeat dies with the process; THIS does not.
set -uo pipefail
ROOT=/home/smarton/tt-metal/.claude/worktrees/ltxperf-tip
LOG="$ROOT/opt/cron.log"
CLAUDE=/home/smarton/.local/bin/claude
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Halt sentinels — a running claude session or the user drops these to stop the loop.
if [[ -e "$ROOT/STOP" ]]; then echo "$(ts) STOP present — halted" >>"$LOG"; exit 0; fi
if [[ -e "$ROOT/DONE" ]]; then echo "$(ts) DONE present — halted" >>"$LOG"; exit 0; fi

# Skip if an interactive session touched its liveness marker in the last 18 min: it owns the laps,
# cron is only the failsafe for when that session is dead. Prevents double-dispatch.
MARK="$ROOT/opt/.session_alive"
if [[ -f "$MARK" ]]; then
  age=$(( $(date +%s) - $(stat -c %Y "$MARK" 2>/dev/null || echo 0) ))
  if (( age < 1080 )); then echo "$(ts) interactive session alive (${age}s) — cron lap skipped" >>"$LOG"; exit 0; fi
fi

echo "$(ts) cron lap firing (no live session)" >>"$LOG"
cd "$ROOT" || exit 1
timeout 1800 "$CLAUDE" -p "$(cat "$ROOT/opt/LOOP_TASK.md")" \
  --dangerously-skip-permissions --model claude-opus-4-8 >>"$LOG" 2>&1
echo "$(ts) cron lap exited rc=$?" >>"$LOG"
