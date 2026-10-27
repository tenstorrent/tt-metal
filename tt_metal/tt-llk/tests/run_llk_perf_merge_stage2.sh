#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Merge gate stage 2, in a leg after stage 1: re-measure this leg's regressions on both commits at 5 code positions.
# Usage: CHIP_ARCH=<arch> THRESHOLD=<x> [MODULE_THRESHOLDS=<m=x,...>] LLK_STAGE2_DEADLINE=<epoch s> run_llk_perf_merge_stage2.sh <baseline dir> <out dir>
set -euo pipefail

BASELINE="$(realpath -m "${1:?usage: run_llk_perf_merge_stage2.sh <baseline dir> <out dir>}")"
OUT="$(realpath -m "${2:?usage: run_llk_perf_merge_stage2.sh <baseline dir> <out dir>}")"
DEADLINE="${LLK_STAGE2_DEADLINE:?LLK_STAGE2_DEADLINE must be an epoch time}"
CAP="${LLK_STAGE2_CAP:-300}"
LLK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(git -C "$LLK" rev-parse --show-toplevel)"
mkdir -p "$OUT"
if [[ ! -f "$BASELINE/baseline.csv" ]]; then
  echo "::warning::No merge baseline for ${CHIP_ARCH}; no stage 2."
  exit 0
fi

COMPARE=(--threshold "${THRESHOLD:?}" --min-cycles 30 --module-threshold "${MODULE_THRESHOLDS:-}")

python3 "$LLK/perf/regression_compare.py" "${COMPARE[@]}" \
  --current "$LLK/perf_data/runs/**/*.csv" --baseline "$BASELINE/**/*.csv" \
  --run-types L1_TO_L1 --arch "$CHIP_ARCH" --report "$OUT/stage1.md" > /dev/null || true
if [[ ! -s "$OUT/stage1.regressions.csv" ]]; then
  echo "No stage-1 regression in this leg."
  exit 0
fi
python3 "$LLK/perf/placement_check.py" nodes --candidates "$OUT/stage1.regressions.csv" \
  --map "$LLK/perf_data/runs/**/*.windowhash" --cap "$CAP" --out "$OUT/nodes.txt"

measure() {
  local left=$(( $3 - $(date +%s) ))
  if (( left < 30 )); then
    echo "::warning::No time left to measure the $1 side."
    return
  fi
  LLK_PERF_NODE_LIST="$OUT/nodes.txt" RUNNER_TEMP="$OUT/tmp-$1" timeout -k 20 "$left" "$2/tests/run_llk_perf_placement.sh" \
    || echo "::warning::Stage 2 on the $1 side did not finish; what it missed is reported as not re-measured."
}

BASE_SHA="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("commit_sha", ""))' "$BASELINE/baseline_metadata.json")"
BASE="$OUT/base"
mkdir -p "$BASE"
git -C "$REPO" fetch -q --depth 1 origin "$BASE_SHA"
git -C "$REPO" archive FETCH_HEAD tt_metal/tt-llk tt_metal/hw tt_metal/hostdevcommon \
  ttnn/cpp/ttnn/operations/experimental | tar -x -C "$BASE"

if [[ ! -x "$BASE/tt_metal/tt-llk/tests/run_llk_perf_placement.sh" ]]; then
  echo "::warning::The baseline commit ${BASE_SHA} predates stage 2; nothing to compare with."
elif ! cmp -s "$LLK/tests/sfpi-version" "$BASE/tt_metal/tt-llk/tests/sfpi-version"; then
  echo "::warning::The baseline commit ${BASE_SHA} pins another SFPI; nothing to compare with."
else
  ln -s "$LLK/tests/sfpi" "$BASE/tt_metal/tt-llk/tests/sfpi"
  # Each side gets half of the time left.
  measure head "$LLK" $(( ($(date +%s) + DEADLINE) / 2 ))
  measure base "$BASE/tt_metal/tt-llk" "$DEADLINE"
fi

mkdir -p "$BASE/tt_metal/tt-llk/perf_data/runs"
python3 "$LLK/perf/placement_check.py" decide "${COMPARE[@]}" --candidates "$OUT/stage1.regressions.csv" \
  --baseline "$BASE/tt_metal/tt-llk/perf_data/runs" --current "$LLK/perf_data/runs" \
  --over-cap "$OUT/nodes.over" --arch "$CHIP_ARCH" --report "$OUT/stage2.md" || true
