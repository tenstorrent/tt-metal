#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Refreshing status table (PASS / HANG? / FAIL / RUN / STALE / PENDING) over the
# outer-loop logs in one log dir.
#
# Usage: watch.sh [log_name] [loop_count]    (REFRESH / STALE_SECS overridable via env)

source "$(dirname "$0")/common.sh" "$@"
REFRESH="${REFRESH:-15}"

while true; do
  scan_log_dir "$LOG_DIR"

  # Rendered to fit the pane: an overflowing page scrolls the header off the top, so only the newest
  # iteration rows that fit are shown. A row wider than the pane wraps onto ceil(len/cols) screen lines.
  cols=$(tput cols); rows=$(tput lines)
  header=(
    "══ STRESS x${LOOP}  $LOG_NAME  iter${INNER_ITERS}  $(date '+%Y-%m-%d %H:%M:%S') ══════════════════"
    "  TT_METAL_HOME=$TT_METAL_HOME"
    "$(printf "  PASS=%d  CRASH=%d  HANG?=%d  FAIL=%d  RUN=%d  PENDING=%d" "$pass" "$crash" "$hang" "$fail" "$running" "$pending")"
    "  ─────────────────────────────────────────────────────────────"
  )
  footer=("" "  refresh: ${REFRESH}s    log dir: $LOG_DIR")
  budget=$((rows - 1))
  for l in "${header[@]}" "${footer[@]}"; do budget=$((budget - (${#l} + cols - 1) / cols - (${#l} == 0))); done
  first=${#details[@]}
  while ((first > 0)); do
    l=${details[first - 1]}
    budget=$((budget - (${#l} + cols - 1) / cols))
    ((budget < 0)) && break
    ((first--))
  done

  clear
  printf '%s\n' "${header[@]}" "${details[@]:first}" "${footer[@]}"
  sleep "$REFRESH"
done
