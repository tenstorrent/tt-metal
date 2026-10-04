#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Launches a whole stress run in ONE detached tmux window with three panes:
#
#   ┌──────────────────────┬──────────────────────┐
#   │ 0  watch.sh          │ 1  stress.sh         │
#   │    (status table)    │    (the pytest loop) │
#   │                      ├──────────────────────┤
#   │                      │ 2  host_stats.sh     │
#   │                      │    (CPU / DRAM)      │
#   └──────────────────────┴──────────────────────┘
#
# Usage:  MODEL=KIMI_K2_7 [TRACE_ID=notrace] [TRIAGE=1 [HANG_SECS=120]] \
#         ./launch.sh [loop_count] [log_name]
#
# Both args are optional: loop_count defaults to 20, and log_name defaults to
# LOG_<date>_<host>_<model>_<commit>_loop_<n>. The resolved log name is printed
# and also echoed into the window's pane titles.

set -u

SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-/data/$USER/tt-metal}"

if [ -z "${MODEL:-}" ]; then
  echo "ERROR: set MODEL to one of: KIMI_K2_7 | GLM5_3 | GEMMA4" >&2
  echo "  e.g.  MODEL=KIMI_K2_7 $0 20" >&2
  exit 1
fi

if [ ! -d "$TT_METAL_HOME" ]; then
  echo "ERROR: TT_METAL_HOME=$TT_METAL_HOME does not exist." >&2
  exit 1
fi

# The scripts run from wherever *this* file lives, but the repo under test (venv,
# test file, PYTHONPATH) is TT_METAL_HOME. Those are usually the same checkout;
# when they are not it is worth saying out loud, because the run then tests code
# you are not editing.
case "$SCRIPTS/" in
  "$TT_METAL_HOME"/*) ;;
  *) echo "NOTE: scripts are in $SCRIPTS but TT_METAL_HOME=$TT_METAL_HOME — the run tests the latter." ;;
esac

LOOP_CNT="${1:-20}"
COMMIT_HASH=$(git -C "$TT_METAL_HOME" rev-parse --short HEAD 2>/dev/null || echo nogit)
DATE=$(date +%Y_%m_%d_%H_%M)
LOG_NAME="${2:-LOG_${DATE}_${HOSTNAME}_${MODEL}_${COMMIT_HASH}_loop_${LOOP_CNT}}"
LOG_DIR="${LOG_ROOT:-/data/$USER}/$LOG_NAME"
mkdir -p "$LOG_DIR" || exit 1

SESSION="${SESSION:-stress_${HOSTNAME}}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$SESSION' already exists." >&2
  echo "  attach:  tmux attach -t $SESSION -r" >&2
  echo "  or kill: tmux kill-session -t $SESSION   (or pass SESSION=<other name>)" >&2
  exit 1
fi

# Each pane's env is put on its own command line rather than relying on inheritance.
# Panes are spawned by the tmux *server*, so what they inherit is the environment
# that server was started with plus the session env — for an already-running server
# that is some older shell's environment, and an override you set for this launch
# silently does not reach the pane. It then runs the default repo / node id and
# looks like it worked.
#
# RUN_ENV goes to both run panes, not just stress.sh: watch.sh sources common.sh too,
# so TT_METAL_HOME and MODEL feed its LOG_DIR, and ITERS_ID feeds INNER_ITERS — the
# "iter=N/M" denominator and the header in the status table.
# TRACE_ID / PREFLIGHT / LOGURU_LEVEL only matter to stress.sh, but are threaded
# uniformly so the panes cannot describe different run points.
# The loop count needs no env: it is positional arg 2 of every script.
# TRIAGE / HANG_SECS arm stress.sh's hang detection. TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES
# is a metal runtime var read by pytest, so it has to reach the pane rather than only this shell.
printf -v RUN_ENV '%q %q' "TT_METAL_HOME=$TT_METAL_HOME" "MODEL=$MODEL"
for v in TRACE_ID MARGIN_ID MESH_ID PRELOAD_ID CHUNKS_ID ITERS_ID STALE_SECS LOGURU_LEVEL PREFLIGHT DS_PERF_IGNORE_POWER \
         TRIAGE HANG_SECS TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES LOG_ROOT RESET_BETWEEN_RUNS TT_SMI \
         HF_MODEL HF_HOME HF_HUB_OFFLINE TT_CACHE_PATH PREFILL_TRACE_DIR; do
  if [ -n "${!v:-}" ]; then
    printf -v quoted '%q' "$v=${!v}"
    RUN_ENV+=" $quoted"
  fi
done

printf -v STRESS_CMD '%s %q %q %q |& tee %q; exit ${PIPESTATUS[0]}' "$RUN_ENV" "$SCRIPTS/stress.sh" "$LOG_NAME" "$LOOP_CNT" "$LOG_DIR/stress.log"
printf -v WATCH_CMD '%s %q %q %q' "$RUN_ENV" "$SCRIPTS/watch.sh" "$LOG_NAME" "$LOOP_CNT"
# log_name only, so the pane appends its 60s snapshots to <log dir>/host_stats.tsv.
# No run env: it watches the box, not a run point.
printf -v HOST_CMD '%s %q %q' "$RUN_ENV" "$SCRIPTS/host_stats.sh" "$LOG_NAME"

run_pane() { printf 'bash -l -c %q' "$1"; }

# Explicit layout: stress split vertically for host, then a full-height status column
# added on the left — deliberately NOT `select-layout tiled`, which reflows the panes
# and renumbers them, so the pane indices would no longer match the order the commands
# were attached in. Panes are
# tracked by pane *id* (%N, stable for the pane's lifetime) for the same reason.
tmux new-session -d -s "$SESSION" -n stress -x 240 -y 60 \
  -E "$(run_pane "$STRESS_CMD")"
P_STRESS=$(tmux list-panes -t "$SESSION:stress" -F '#{pane_id}')
# `failed`, not `on`: a pane whose command exits non-zero is kept so its error is still on screen,
# while a clean end-of-loop pane closes as before. Without this a preflight rejection — the stale
# node id case — kills the pane in ~2s and the launch looks like it did nothing at all. Set here,
# right after the pane is created, so it is in place before the preflight can finish.
tmux set-option -t "$SESSION:stress" -w remain-on-exit failed 2>/dev/null || true
P_HOST=$(tmux split-window -v -P -F '#{pane_id}' -t "$P_STRESS" "$(run_pane "$HOST_CMD")")
P_WATCH=$(tmux split-window -h -b -f -P -F '#{pane_id}' -t "$P_STRESS" "$(run_pane "$WATCH_CMD")")

# Titled panes so an attached window is readable without guessing which is which.
tmux set-option -t "$SESSION" -w pane-border-status top
tmux set-option -t "$SESSION" -w pane-border-format ' #{pane_index}: #{pane_title} '
tmux select-pane -t "$P_STRESS" -T "stress $MODEL x$LOOP_CNT"
tmux select-pane -t "$P_WATCH" -T "status $LOG_NAME"
tmux select-pane -t "$P_HOST" -T "host cpu/dram"
tmux select-pane -t "$P_STRESS"

cat <<EOF
launched tmux session '$SESSION' (1 window, 3 panes)
  MODEL=$MODEL${TRACE_ID:+  TRACE_ID=$TRACE_ID}  LOOP=$LOOP_CNT
  LOG_NAME=$LOG_NAME
  log dir=$LOG_DIR
  output=$LOG_DIR/stress.log

attach (read-only):  tmux attach -t $SESSION -r
zoom one pane:       ctrl-b z      switch panes: ctrl-b <arrow>
kill the run:        tmux kill-session -t $SESSION
EOF
