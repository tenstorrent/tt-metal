#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Tails the newest log_NN file in the log dir.
#
# Usage: tail.sh [log_name] [loop_count]    (REFRESH overridable via env)

source "$(dirname "$0")/common.sh" "$@"
REFRESH="${REFRESH:-30}"

while true; do
  clear
  echo "══ TAIL LATEST LOG  $(date '+%H:%M:%S')  refresh ${REFRESH}s ═══════════════════════"
  LATEST=$(ls -1t "$LOG_DIR"/log_?? 2>/dev/null | head -1)
  if [ -n "$LATEST" ]; then
    echo "  file: $LATEST"
    echo "  size: $(stat -c %s "$LATEST") bytes"
    echo "  mtime: $(stat -c '%y' "$LATEST")"
    echo "  ──────────────────────────────────────────────"
    tail -10 "$LATEST"
  else
    echo "  (no log_NN file yet — waiting)"
  fi
  sleep "$REFRESH"
done
