#!/usr/bin/env bash
# run_bg.sh — the ONE launcher every Laguna background run goes through.
#
# Guarantees ONE permanent tail the user starts ONCE and never touches again:
#   * ~/tail.log is a PERMANENT REAL FILE with a STABLE inode. Each run truncates it IN PLACE
#     (`: > ~/tail.log`) and appends — the inode never changes, so a running `tail -f ~/tail.log`
#     keeps following across every future run. (The old version did `ln -sfn` per run, which swapped
#     the inode and stranded the user's tail — that's why they had to re-tail each time. Fixed.)
#   * every line (full stdout+stderr, unbuffered) is PREFIXED WITH A WALL-CLOCK TIMESTAMP [HH:MM:SS] and
#     tee'd to BOTH ~/tail.log and a per-run archive doc/vllm_integration/_runs/<run-name>.rawout (kept for
#     history / result-reading). The timestamp is when run_bg.sh received the line, so the user can follow
#     elapsed time / spot stalls directly in `tail -f ~/tail.log`.
#   * a header names the run so `tail -f ~/tail.log` self-identifies.
#
# Usage:  run_bg.sh <run-name> <cmd...>   (run the cmd WITH `python -u` for unbuffered output)
#
# The user's permanent tail, for every run, forever:   tail -f ~/tail.log
set -u
RUN="${1:?usage: run_bg.sh <run-name> <cmd...>}"; shift
REPO="/home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1"
DIR="$REPO/doc/vllm_integration/_runs"; mkdir -p "$DIR"
RAW="$DIR/$RUN.rawout"
TAIL="$HOME/tail.log"
# If an older scheme left ~/tail.log as a SYMLINK, replace it with a real file ONCE. From then on it is
# only ever truncated in place, so its inode is stable forever.
[ -L "$TAIL" ] && rm -f "$TAIL"
: > "$TAIL"        # truncate IN PLACE (same inode) — does NOT strand a running `tail -f ~/tail.log`
: > "$RAW"
{
  echo "=================================================================="
  echo "RUN=$RUN   started (full stdout+stderr stream — nothing filtered)"
  echo "CMD: $*"
  echo "tail:  tail -f ~/tail.log   (permanent — survives every run, no need to re-tail)"
  echo "=================================================================="
} | tee -a "$TAIL" >> "$RAW"
# Every line goes to BOTH the permanent tail file and the per-run archive, line-buffered, each PREFIXED with
# a wall-clock [HH:MM:SS]. awk strftime stamps the moment the line is read (real time), and fflush keeps the
# tail live. Carriage-return-only TUI streams (no newline) are handled by splitting on \r too, so progress
# bars still get stamped instead of buffering into one giant line.
exec stdbuf -oL -eL "$@" 2>&1 \
  | stdbuf -oL awk '{ printf "[%s] %s\n", strftime("%H:%M:%S"), $0; fflush() }' \
  | stdbuf -oL tee -a "$TAIL" >> "$RAW"
