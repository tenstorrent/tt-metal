#!/bin/bash
# Refresh 60s. Status table for 20-run iter200 stress.
DIR=/home/ubuntu/devs/tt-metal/stress_x20_iter200

while true; do
  clear
  echo "══ STRESS x20  regular+CF8 25600 iter200  $(date '+%Y-%m-%d %H:%M:%S') ══════════════════"
  echo ""

  pass=0; fail=0; hang=0; running=0; pending=0
  details=()

  for i in $(seq 1 20); do
    N=$(printf "%02d" "$i")
    NEXT=$(printf "%02d" $((i + 1)))
    f="$DIR/log_$N"
    if [ ! -f "$f" ]; then
      ((pending++))
      continue
    fi
    if grep -qE 'smoke test passed|^=+.*1 passed' "$f" 2>/dev/null; then
      elapsed=$(grep -oE '[0-9]+\.[0-9]+s \([0-9:]+\)' "$f" | tail -1)
      details+=("  $N: PASS  $elapsed")
      ((pass++))
    elif grep -qE '^=+.*(1 failed|1 error)' "$f" 2>/dev/null; then
      details+=("  $N: FAIL")
      ((fail++))
    else
      iter=$(grep -c 'Starting iteration:' "$f" 2>/dev/null)
      layer=$(grep -oE 'forward_layer_[0-9]+_(start|end)' "$f" 2>/dev/null | tail -1)
      mtime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
      now=$(date +%s)
      idle=$((now - mtime))

      if [ -f "$DIR/log_$NEXT" ]; then
        details+=("  $N: HANG?  iter=$iter/200  $layer")
        ((hang++))
      elif [ "$idle" -gt 240 ]; then
        details+=("  $N: STALE ${idle}s  iter=$iter/200  $layer")
        ((running++))
      else
        details+=("  $N: RUN    iter=$iter/200  $layer  (idle ${idle}s)")
        ((running++))
      fi
    fi
  done

  printf "  PASS=%d  HANG?=%d  FAIL=%d  RUN=%d  PENDING=%d\n" "$pass" "$hang" "$fail" "$running" "$pending"
  echo "  ─────────────────────────────────────────────────────────────"
  printf '%s\n' "${details[@]}"

  echo ""
  echo "  refresh: 60s    log dir: $DIR"
  sleep 60
done
