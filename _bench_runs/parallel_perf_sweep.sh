#!/usr/bin/env bash
# Parallel perf trace sweep across BH devices.
# Reads (label, env_overrides) pairs from a TSV on stdin (label\tENV=val ENV=val ...)
# and launches each line on a separate device via TT_VISIBLE_DEVICES.
# Aggregates "Per-call avg: NN.NN ms" from each output.
#
# Usage:
#   { echo -e "label1\tPI0_VLM_MINIMAL_CFG=4,8,8,1,8";
#     echo -e "label2\tPI0_VLM_MINIMAL_CFG=4,8,4,1,8"; } | \
#   bash _bench_runs/parallel_perf_sweep.sh
#
# Each line gets a separate Python subprocess; -P <N> caps concurrency
# (default 8; 32 max — but 32 may OOM host RAM).

set -uo pipefail

ROOT="${ROOT:-/home/tt-admin/sdawle/pi0/tt-metal}"
CONCURRENCY="${CONCURRENCY:-8}"
DENOISE_STEPS="${DENOISE_STEPS:-5}"
LOG_DIR="${LOG_DIR:-$ROOT/_bench_runs/perf_sweep_$(date +%Y%m%dT%H%M%SZ)}"
mkdir -p "$LOG_DIR"

# Read all input lines first so we can index them
mapfile -t LINES < <(cat)
NUM=${#LINES[@]}
echo "Running $NUM configs, concurrency=$CONCURRENCY, logs → $LOG_DIR"
echo

# Number lines and feed to xargs
for i in "${!LINES[@]}"; do
  echo -e "$i\t${LINES[$i]}"
done | xargs -P "$CONCURRENCY" -d '\n' -I {} bash -c '
  line="{}"
  idx=$(echo "$line" | cut -f1)
  rest=$(echo "$line" | cut -f2-)
  label=$(echo -e "$rest" | cut -f1)
  envs=$(echo -e "$rest" | cut -f2-)

  device=$(( idx % 32 ))
  log="'"$LOG_DIR"'/$(printf "%02d" $idx)_${label}.log"

  cd "'"$ROOT"'"
  source python_env/bin/activate

  TT_VISIBLE_DEVICES=$device \
  PI0_EXPERT_MM_LOFI=1 PI0_ROPE_TABLES_L1=1 PI0_MM_SWEEP_V2=1 \
  PI0_DENOISE_MM_TUNE=1 PI0_PREFILL_MM_TUNE=1 PI0_UPSTREAM_MASKS=1 \
  QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1 QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1 \
  PI0_NUM_CAMERAS=3 PI0_VLM_CHUNK_SIZE=1024 PI0_VLM_MLP_BF8_OUT=1 PI0_VLM_MLP_MINIMAL=1 \
  PI05_NUM_DENOISE_STEPS='"$DENOISE_STEPS"' \
  PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream \
  TT_METAL_CACHE='"$ROOT"'/.tt_metal_cache TT_METAL_HOME='"$ROOT"' PYTHONPATH='"$ROOT"' \
  $envs \
    timeout 600 pytest -xvs models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_trace.py > "$log" 2>&1

  ms=$(grep -oE "Per-call avg:\s+[0-9.]+ ms" "$log" | head -1 | grep -oE "[0-9.]+")
  if [[ -n "$ms" ]]; then
    printf "  %s (dev=%d)  →  %s ms\n" "$label" "$device" "$ms"
  else
    fail=$(grep -oE "RuntimeError|TT_THROW|TT_FATAL" "$log" | head -1)
    printf "  %s (dev=%d)  →  ❌ %s — see %s\n" "$label" "$device" "${fail:-UNKNOWN}" "$log"
  fi
'

echo
echo "============================================================"
echo "  LEADERBOARD"
echo "============================================================"
{
  for log in "$LOG_DIR"/*.log; do
    base=$(basename "$log" .log)
    ms=$(grep -oE "Per-call avg:\s+[0-9.]+ ms" "$log" | head -1 | grep -oE "[0-9.]+")
    if [[ -n "$ms" ]]; then
      printf "%s\t%s\n" "$ms" "$base"
    else
      printf "999\t%s ❌\n" "$base"
    fi
  done
} | sort -n | awk -F'\t' '{ printf "  %s ms  %s\n", $1, $2 }'
echo "============================================================"
echo "Logs: $LOG_DIR"
