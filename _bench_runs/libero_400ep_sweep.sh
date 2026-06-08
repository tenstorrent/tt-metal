#!/usr/bin/env bash
# 400-episode LIBERO sweep on the MINIMAL path (chunk=1024 + bf8_out + minimal_matmul).
# 40 tasks (4 suites × 10) × 10 init states = 400 episodes.
# Sharded across 32 BH devices via TT_VISIBLE_DEVICES; xargs -P 32 caps concurrency.
#
# Each task launches an independent rollout process targeting one logical TT
# device. With 40 tasks > 32 devices, the first wave of 32 starts immediately
# and the remaining 8 wait for slots to free up.
#
# Output: per-task log under $LOG_DIR/<suite>_<task>.log
# Aggregation grep at the end.

set -uo pipefail

ROOT="${ROOT:-/home/tt-admin/sdawle/pi0/tt-metal}"
LOG_DIR="${LOG_DIR:-$ROOT/_bench_runs/libero_400ep_$(date +%Y%m%dT%H%M%SZ)}"
mkdir -p "$LOG_DIR"

echo "Logs → $LOG_DIR"
echo "Devices visible: $(seq 0 31 | wc -l)"

# Generate the 40 (suite, task) pairs as one per line
TASKS=$(mktemp)
trap "rm -f $TASKS" EXIT
for suite in libero_spatial libero_object libero_goal libero_10; do
  for idx in $(seq 0 9); do
    echo "$suite:$idx"
  done
done > "$TASKS"

# Launcher: each line of TASKS becomes one rollout invocation.
# $1 = SUITE:IDX
# We index lines so we can rotate device assignment 0..31.
nl -ba "$TASKS" | xargs -P 32 -L 1 bash -c '
  line_no="$0"
  task="$1"
  suite="${task%%:*}"
  idx="${task##*:}"
  device=$(( (line_no - 1) % 32 ))
  log="'"$LOG_DIR"'/${suite}_t${idx}.log"

  cd "'"$ROOT"'"
  source python_env/bin/activate

  # All pi0.5 perf flags come from _bench_runs/pi05_production.env (single source of truth).
  # Source it here so the per-task subshell inherits the exports.
  source '"$ROOT"'/_bench_runs/pi05_production.env

  TT_VISIBLE_DEVICES=$device \
  TT_METAL_CACHE='"$ROOT"'/.tt_metal_cache \
  TT_METAL_HOME='"$ROOT"' \
  PYTHONPATH='"$ROOT"':/home/tt-admin/pi05_cache/libero_repo \
  LIBERO_REPO_PATH=/home/tt-admin/pi05_cache/libero_repo \
  PI0_TOKENIZER_PATH=/home/tt-admin/pi05_cache/tokenizer/paligemma_tokenizer.model \
  MUJOCO_GL=osmesa \
  python models/experimental/pi0_5/eval/libero_rollout.py \
    --checkpoint /home/tt-admin/pi05_cache/pi05_libero_upstream \
    --backend ttnn \
    --device-id 0 \
    --tasks "${suite}:${idx}" \
    --num-episodes 10 \
    --steps-sweep 5 \
    --action-horizon 10 \
    --state-in-prompt false \
    > "$log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "❌ ${suite}:${idx} (dev=$device) → exit $rc — see $log"
  else
    # Count successes from the per-task summary line:
    #   "  libero_spatial     task  0:  X/10  avg_steps= …"
    s=$(grep -oE "task[[:space:]]+${idx}:[[:space:]]+[0-9]+/10" "$log" | head -1 | grep -oE "[0-9]+/10" | head -1)
    echo "✓ ${suite}:${idx} (dev=$device) → ${s:-?}"
  fi
'

echo
echo "=================================================================="
echo "                   AGGREGATE  (400-ep MINIMAL sweep)"
echo "=================================================================="

total_succ=0
total_attempted=0
for suite in libero_spatial libero_object libero_goal libero_10; do
  suite_succ=0
  suite_attempted=0
  echo
  echo "  $suite:"
  for idx in $(seq 0 9); do
    log="$LOG_DIR/${suite}_t${idx}.log"
    if [[ -f "$log" ]]; then
      # Extract the "N/10" fragment from "task  N:  S/10" and strip the trailing "/10".
      # Earlier version used `grep -oE "^[0-9]+"` which never matched because the
      # captured fragment starts with "task", not a digit — silently gave s=0.
      sf=$(grep -oE "task[[:space:]]+${idx}:[[:space:]]+[0-9]+/10" "$log" | head -1 | grep -oE "[0-9]+/10" | head -1)
      s=${sf%/10}
      s=${s:-0}
      attempted=$(grep -c "ep [0-9]\+: success" "$log" 2>/dev/null || echo 0)
      printf "    task %d:  %d/10   (%d eps recorded)\n" "$idx" "$s" "$attempted"
      suite_succ=$((suite_succ + s))
      suite_attempted=$((suite_attempted + attempted))
    else
      printf "    task %d:  no log\n" "$idx"
    fi
  done
  printf "    SUITE TOTAL:  %d/%d\n" "$suite_succ" "$suite_attempted"
  total_succ=$((total_succ + suite_succ))
  total_attempted=$((total_attempted + suite_attempted))
done

echo
echo "  --------------------------------------------------------------"
printf "  GRAND TOTAL:  %d / %d  (%.1f%%)\n" "$total_succ" "$total_attempted" \
  "$(awk "BEGIN{print 100*$total_succ/($total_attempted+0.0001)}")"
echo "=================================================================="
echo "Logs: $LOG_DIR"
