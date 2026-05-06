#!/bin/bash
# Tails the latest log_NN file in stress_x20_iter200 — refresh 30s.
DIR=/home/ubuntu/devs/tt-metal/stress_x20_iter200

while true; do
  clear
  echo "══ TAIL LATEST LOG  $(date '+%H:%M:%S')  refresh 30s ═══════════════════════"
  LATEST=$(ls -1t "$DIR"/log_?? 2>/dev/null | head -1)
  if [ -n "$LATEST" ]; then
    echo "  file: $LATEST"
    echo "  size: $(stat -c %s "$LATEST") bytes"
    echo "  mtime: $(stat -c '%y' "$LATEST")"
    echo "  ──────────────────────────────────────────────"
    tail -10 "$LATEST"
  else
    echo "  (no log_NN file yet — waiting)"
  fi
  sleep 30
done
