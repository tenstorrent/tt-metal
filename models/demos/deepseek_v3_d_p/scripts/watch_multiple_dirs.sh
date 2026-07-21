#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Status table over several log dirs at once — one stacked block per dir.
#
# Usage: watch_multiple_dirs.sh [log_name ...]    (REFRESH / LOOP / STALE_SECS via env)

# Don't forward positional args: here every arg is a log_name (handled below as
# DIRS), not common.sh's <log_name> [loop_count]. LOOP comes from the env.
source "$(dirname "$0")/common.sh"
REFRESH="${REFRESH:-15}"

# Each positional arg is a log_name; default to the single common LOG_NAME.
DIRS=("$@")
[ ${#DIRS[@]} -eq 0 ] && DIRS=("$LOG_NAME")

while true; do
  clear
  echo "══ STRESS MONITOR  iter${INNER_ITERS}  $(date '+%Y-%m-%d %H:%M:%S') ══════════════════════════════"

  for name in "${DIRS[@]}"; do
    dir="/data/$USER/$name"
    echo "── Folder: $name ────────────────────────────────────────────────"

    scan_log_dir "$dir"

    printf "  PASS=%d  HANG?=%d  FAIL=%d  RUN=%d  PENDING=%d\n" "$pass" "$hang" "$fail" "$running" "$pending"
    printf '%s\n' "${details[@]}"
    echo ""
  done

  echo "  refresh: ${REFRESH}s"
  sleep "$REFRESH"
done
