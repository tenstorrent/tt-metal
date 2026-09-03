#!/bin/bash
# galaxy-kit node launcher — runs ON the galaxy node via:
#   srun --overlap --jobid <id> bash <BASE>/galaxy_launch.sh <chips>
# <chips> = "all" (0-31) | comma list | a count N (chips 0..N-1).
# EXABOX.md §7 walls honored: ulimit raised first (wall 4); ONE upfront
# tt-smi -r for the whole node, marker-guarded (wall 2 allows exactly this);
# workers never reset anything.
ulimit -u "$(ulimit -Hu)" 2>/dev/null
set -u
BASE=${LK_BASE:-/data/nkapre/craq-laneLK}
CHIPS=${1:?chips}
if [ "$CHIPS" = "all" ]; then CHIPS=$(seq -s, 0 31);
elif [[ "$CHIPS" =~ ^[0-9]+$ ]] && [ "$CHIPS" -le 32 ] && [[ "$CHIPS" != *,* ]]; then
  CHIPS=$(seq -s, 0 $((CHIPS-1)))
fi
echo "LK-GALAXY host=$(hostname) chips=$CHIPS reps=${LK_REPS:-5} batch=${LK_BATCH:-0}x${LK_BATCH_OPS:-8} $(date -u +%FT%TZ)"
mkdir -p "$BASE/results" "$BASE/wlogs" "$BASE/claims"
if [ ! -f "$BASE/results/.reset-done-$(hostname)" ]; then
  "$BASE/venv/bin/tt-smi" -r > "$BASE/results/tt-smi-reset-$(hostname).log" 2>&1
  echo "one-shot tt-smi -r rc=$?"
  touch "$BASE/results/.reset-done-$(hostname)"
fi
PIDS=()
IFS=',' read -ra CL <<< "$CHIPS"
for c in "${CL[@]}"; do
  LK_BASE="$BASE" "$BASE/venv/bin/python" "$BASE/worker.py" --chip "$c" \
    > "$BASE/wlogs/chip$c.log" 2>&1 &
  PIDS+=($!)
done
rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=1; done
echo "LK-GALAXY-DONE rc=$rc"
exit $rc
