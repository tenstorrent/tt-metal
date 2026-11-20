#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Merge gate stage 2, in a leg after stage 1: re-measure this leg's regressions on both commits at 5 code positions.
# Usage: CHIP_ARCH=<arch> THRESHOLD=<x> [MODULE_THRESHOLDS=<m=x,...>] LLK_STAGE2_DEADLINE=<epoch s>
#        [LLK_STAGE2_BASE_REF=<sha>] run_llk_perf_merge_stage2.sh <baseline dir> <out dir>
set -euo pipefail

BASELINE="$(realpath -m "${1:?usage: run_llk_perf_merge_stage2.sh <baseline dir> <out dir>}")"
OUT="$(realpath -m "${2:?usage: run_llk_perf_merge_stage2.sh <baseline dir> <out dir>}")"
DEADLINE="${LLK_STAGE2_DEADLINE:?LLK_STAGE2_DEADLINE must be an epoch time}"
LLK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(git -C "$LLK" rev-parse --show-toplevel)"
mkdir -p "$OUT"
if [[ ! -f "$BASELINE/baseline.csv" ]]; then
  echo "::warning::No merge baseline for ${CHIP_ARCH}; no stage 2."
  exit 0
fi

# The gate checks this against its own baseline: a leg that read another one does not count.
python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("commit_sha", ""))' \
  "$BASELINE/baseline_metadata.json" > "$OUT/stage2.base"
# Stage 2 measures the queue base, the commit this PR merges onto. A stale warehouse baseline then only adds
# candidates, which stage 2 does not reproduce.
BASE_SHA="${LLK_STAGE2_BASE_REF:-$(cat "$OUT/stage2.base")}"

COMPARE=(--threshold "${THRESHOLD:?}" --min-cycles 30 --module-threshold "${MODULE_THRESHOLDS:-}")

python3 "$LLK/perf/regression_compare.py" "${COMPARE[@]}" \
  --current "$LLK/perf_data/runs/**/*.csv" --baseline "$BASELINE/**/*.csv" \
  --run-types L1_TO_L1 --arch "$CHIP_ARCH" --report "$OUT/stage1.md" > /dev/null || true
if [[ ! -f "$OUT/stage1.verdict.json" ]]; then
  echo "::warning::The stage-1 comparison of this leg did not finish; no stage 2."
  exit 0
fi
if [[ ! -s "$OUT/stage1.regressions.csv" ]]; then
  echo "No stage-1 regression in this leg."
  # An empty result still tells the gate this leg's stage 2 ran.
  echo "[]" > "$OUT/stage2.json"
  exit 0
fi
python3 "$LLK/perf/placement_check.py" nodes --candidates "$OUT/stage1.regressions.csv" \
  --map "$LLK/perf_data/runs/**/*.windowhash" --cap 1000000 --out "$OUT/nodes.txt"

# measure <side> <tree> <deadline> <part>: the part's tests at the 5 positions, kept as perf_data/runs/p<shift>-<part>.
measure() {
  local left=$(( $3 - $(date +%s) ))
  if (( left < 30 )); then
    echo "::warning::No time left to measure part $4 on the $1 side."
    return
  fi
  LLK_PERF_NODE_LIST="$OUT/nodes.$4" RUNNER_TEMP="$OUT/tmp-$1-$4" timeout -k 20 "$(( left - 20 ))" "$2/tests/run_llk_perf_placement.sh" \
    || echo "::warning::Stage 2 part $4 on the $1 side did not finish; what it missed is reported as not re-measured."
  for d in "$2"/perf_data/runs/p{0,4,20,36,52}; do
    [[ -d $d ]] && mv "$d" "$d-$4"
  done
  # Both sides run the PR's node ids; a test the base does not have (renamed, new) is skipped there.
  local listed found
  listed=$(wc -l < "$OUT/nodes.$4")
  found=$(python3 -c 'import csv, glob, sys
print(len({r["nodeid"] for f in glob.glob(sys.argv[1] + "/**/*.windowhash", recursive=True) for r in csv.DictReader(open(f))}))' \
    "$2/perf_data/runs/p0-$4" 2>/dev/null || echo 0)
  if [[ $1 == base ]] && (( found < listed )); then
    echo "::warning::The base ran ${found} of ${listed} listed tests in part $4 (renamed or new tests?); the rest show as not measured on the base."
  fi
}

BASE="$OUT/base"
mkdir -p "$BASE"
git -C "$REPO" fetch -q --depth 1 origin "$BASE_SHA"
git -C "$REPO" archive FETCH_HEAD tt_metal/tt-llk tt_metal/hw tt_metal/hostdevcommon tt_metal/sfpi-version \
  ttnn/cpp/ttnn/operations/experimental | tar -x -C "$BASE"

# No stage2.json from here on: the gate report counts this leg as missing.
if [[ ! -x "$BASE/tt_metal/tt-llk/tests/run_llk_perf_placement.sh" ]]; then
  echo "::warning::The baseline commit ${BASE_SHA} predates stage 2; nothing to compare with."
  exit 0
elif [[ ! -f "$BASE/tt_metal/tt-llk/tests/sfpi-version" ]]; then
  echo "::error::The baseline commit ${BASE_SHA} has no SFPI pin in the archive; nothing to compare with."
  exit 0
elif ! cmp -s "$LLK/tests/sfpi-version" "$BASE/tt_metal/tt-llk/tests/sfpi-version"; then
  echo "::warning::The baseline commit ${BASE_SHA} pins another SFPI; nothing to compare with."
  exit 0
fi
ln -s "$LLK/tests/sfpi" "$BASE/tt_metal/tt-llk/tests/sfpi"
# No test limit: stage 2 measures until the deadline, and the report lists what it did not measure on both sides.
# The largest slowdowns go first, as a part of their own on both sides, so a leg that runs out of time
# still has results for them. Within a part, each side gets half of the time left.
FIRST="${LLK_STAGE2_FIRST:-10}"
head -n "$FIRST" "$OUT/nodes.txt" > "$OUT/nodes.1"
tail -n +"$(( FIRST + 1 ))" "$OUT/nodes.txt" > "$OUT/nodes.2"
for part in 1 2; do
  [[ -s "$OUT/nodes.$part" ]] || continue
  measure head "$LLK" $(( ($(date +%s) + DEADLINE) / 2 )) "$part"
  measure base "$BASE/tt_metal/tt-llk" "$DEADLINE" "$part"
done

mkdir -p "$BASE/tt_metal/tt-llk/perf_data/runs"
python3 "$LLK/perf/placement_check.py" decide "${COMPARE[@]}" --candidates "$OUT/stage1.regressions.csv" \
  --baseline "$BASE/tt_metal/tt-llk/perf_data/runs" --current "$LLK/perf_data/runs" \
  --over-cap "$OUT/nodes.over" --arch "$CHIP_ARCH" --report "$OUT/stage2.md" || true
