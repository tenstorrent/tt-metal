#!/bin/bash
# Shard the rms_norm_ttnn golden cartesian across eval_test_runner.sh invocations.
#
# The whole-directory run does not fit a 10-minute tool call (120 960 collected
# cells; the precompile pass alone is >30 min on 15 620 unique programs), so this
# drives one shard at a time with `-k <shape ids>` and stops before a wall-clock
# budget so the caller stays inside its ceiling. State is the presence of each
# shard's output dir, so re-invoking resumes where the last call stopped.
#
# Usage: scripts/verifier_golden_shards.sh <base_out_dir> <budget_seconds> [shard_index ...]
#   with no shard indices, runs every shard that has no results yet.

set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

BASE="${1:?usage: verifier_golden_shards.sh <base_out_dir> <budget_seconds> [idx ...]}"
BUDGET="${2:?}"
shift 2

# One entry per shard: the `-k` selector. Shapes are grouped so a shard's cell
# count x its per-cell cost lands inside the budget: the wide shapes (4096 /
# 8192 / 1024x1024) are 1-2 per shard, the small ones 3-4.
SHARDS=(
  "1x1x32x64 or 1x1x64x128 or 4x8x32x256"
  "2x4x128x512 or 1x1x2048x256"
  "4x1x512x512 or 1x1x32x50"
  "1x1x64x17 or 4x8x32x47 or 2x1x128x100"
  "1x1x17x64 or 1x1x50x128 or 4x8x47x256"
  "1x1x17x50 or 2x1x100x47"
  "1x1x32x4096"
  "1x1x32x8192"
  "1x1x128x4096 or 2x1x64x4096"
  "1x32x128 or 4x128x512"
  "2x512x1024"
  "1x32x4096 or 1x32x8192"
  "1x32x50 or 4x128x47 or 1x17x128"
  "32x64 or 128x512"
  "1024x1024"
  "32x4096 or 128x8192"
  "32x17 or 128x100 or 17x64"
  "not test_op or test_op_loose"
)

if [[ $# -gt 0 ]]; then
  TODO=("$@")
else
  TODO=()
  for i in "${!SHARDS[@]}"; do
    [[ -f "${BASE}/shard_${i}/test_results.json" ]] || TODO+=("$i")
  done
fi

START=$(date +%s)
mkdir -p "$BASE"
RAN=0
for i in "${TODO[@]}"; do
  ELAPSED=$(( $(date +%s) - START ))
  if (( RAN > 0 && ELAPSED > BUDGET )); then
    echo "SHARDS: budget ${BUDGET}s reached after ${ELAPSED}s — stopping (resume by re-invoking)"
    break
  fi
  SEL="${SHARDS[$i]}"
  OUT="${BASE}/shard_${i}"
  echo "SHARDS: === shard ${i} === -k \"${SEL}\""
  rm -rf "$OUT"
  eval/eval_test_runner.sh --no-precompile eval/golden_tests/rms_norm_ttnn "$OUT" -k "$SEL" \
    2>&1 | grep -E "EVAL_RUNNER: [0-9]+/|EVAL_RUNNER: (HANG|FAIL|ERROR)|passed|failed" | tail -4
  RAN=$((RAN + 1))
done

DONE=0
for i in "${!SHARDS[@]}"; do
  [[ -f "${BASE}/shard_${i}/test_results.json" ]] && DONE=$((DONE + 1))
done
echo "SHARDS: ${DONE}/${#SHARDS[@]} shards have results"
