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
  clear
  echo "══ STRESS x${LOOP}  $LOG_NAME  iter${INNER_ITERS}  $(date '+%Y-%m-%d %H:%M:%S') ══════════════════"
  echo ""

  scan_log_dir "$LOG_DIR"

  printf "  PASS=%d  HANG?=%d  FAIL=%d  RUN=%d  PENDING=%d\n" "$pass" "$hang" "$fail" "$running" "$pending"
  echo "  ─────────────────────────────────────────────────────────────"
  printf '%s\n' "${details[@]}"

  echo ""
  echo "  refresh: ${REFRESH}s    log dir: $LOG_DIR"
  sleep "$REFRESH"
done
