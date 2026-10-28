#!/bin/bash
# Replay one merge-gate report on a reserved board with the new design:
# stack fixes + stage-2 harness on both commits, stage 1 at 2 % / 30 cycles on this board,
# then stage 2 (5 code positions + loop fingerprint) for the stage-1 regressions.
#
# Usage: replay_case.sh <arch> <case id> <base sha12> <head sha12> [modules]
# modules: comma-separated test modules to measure in stage 1 (the ones the gate report flagged); blank = whole suite.
# Trees are the branches perf-trial/replay/<sha12> (commit + stack + harness).
set -uo pipefail
export PATH="${REPLAY_ROOT:-/tmp/replay}/venv/bin:$HOME/.local/bin:$PATH"
ARCH=$1 CASE=$2 BASE=$3 HEAD=$4 MODULES=${5:-} RUN_TYPES=${6:-L1_TO_L1} CAP=900
[[ $MODULES == - ]] && MODULES=""
W=${REPLAY_ROOT:-/tmp/replay}
REPO=$W/repo
OUT=$W/out/$CASE
mkdir -p "$OUT"
exec > >(tee -a "$OUT/log.txt") 2>&1
echo "=== $(date -u +%FT%TZ) case $CASE arch $ARCH base $BASE head $HEAD modules ${MODULES:-all}"
echo "running" > "$OUT/status"

if [[ ! -d $REPO ]]; then
  git clone -q --filter=blob:none --no-checkout https://github.com/tenstorrent/tt-metal "$REPO"
fi

tree() {  # tree <sha12> -> path of a sparse worktree with sfpi installed
  local T=$W/trees/$1
  if [[ ! -f $T/.ready ]]; then
    rm -rf "$T"
    git -C "$REPO" fetch -q origin "perf-trial/replay/$1:refs/heads/replay-$1" || return 1
    git -C "$REPO" worktree prune
    git -C "$REPO" worktree add -q --no-checkout "$T" "replay-$1" || return 1
    git -C "$T" sparse-checkout set --no-cone /tt_metal/tt-llk/ /tt_metal/hw/ /tt_metal/hostdevcommon/ \
      /ttnn/cpp/ttnn/operations/experimental/ /tt_metal/sfpi-info.sh /tt_metal/sfpi-version || return 1
    git -C "$T" checkout -q || return 1
    (cd "$T/tt_metal/tt-llk/tests" && ./setup_testing_env.sh > /dev/null) || return 1
    touch "$T/.ready"
  fi
  echo "$T/tt_metal/tt-llk"
}

measure_suite() {  # measure_suite <llk dir> <tag>
  local L=$1
  rm -rf "$L/perf_data/runs/$2"
  export SPEED_OF_LIGHT=false LLK_PERF_RUN_TYPES=$RUN_TYPES TT_LLK_DISABLE_ASSERTS=1 LLK_PERF_EXITFIRST=false \
    PERF_RUN_TAG=$2 RUNNER_TEMP=$W/tmp/$CASE-$2
  if [[ -z $MODULES ]]; then
    (cd "$L" && ./tests/run_llk_perf_${ARCH}.sh 1 1 > "$OUT/$2.pytest.log" 2>&1)
  else
    local files=() m
    for m in ${MODULES//,/ }; do files+=("$m.py"); done
    (cd "$L/tests/python_tests" && mkdir -p perf_data && {
      pytest -q --override-ini=log_cli=false --compile-producer -n 10 -m "perf and not accuracy" --timeout=60 "${files[@]}"
      pytest -q --override-ini=log_cli=false --compile-consumer --dist loadgroup -n 15 -m "perf and not accuracy" --timeout=60 "${files[@]}"
    } > "$OUT/$2.pytest.log" 2>&1)
  fi
  echo "measured $2: $(find "$L/perf_data/runs/$2" -name '*.csv' | wc -l) csv"
}

LB=$(tree "$BASE") || { echo "failed" > "$OUT/status"; echo "base tree failed"; exit 1; }
LH=$(tree "$HEAD") || { echo "failed" > "$OUT/status"; echo "head tree failed"; exit 1; }
echo "trees ready $(date -u +%T)"

T0=$(date +%s)
measure_suite "$LB" "s1b-$CASE"
measure_suite "$LH" "s1h-$CASE"
echo "stage1_seconds=$(( $(date +%s) - T0 ))" > "$OUT/timing"
for T in "$LB/perf_data/runs/s1b-$CASE" "$LH/perf_data/runs/s1h-$CASE"; do
  if [[ -z $(find "$T" -name '*.csv' 2>/dev/null | head -1) ]]; then
    echo "failed" > "$OUT/status"; echo "stage 1 measured nothing in $T"; exit 1
  fi
done

python3 "$LH/perf/regression_compare.py" --threshold 0.02 --min-cycles 30 --module-threshold "" \
  --current "$LH/perf_data/runs/s1h-$CASE/**/*.csv" --baseline "$LB/perf_data/runs/s1b-$CASE/**/*.csv" \
  --run-types L1_TO_L1 --arch "$ARCH" --report "$OUT/stage1.md" > /dev/null
echo "stage 1: $(sed -n 's/.*\*\*\([0-9]*\) regressed point.*/\1/p' "$OUT/stage1.md" | head -1) regressed points"

if [[ -s $OUT/stage1.regressions.csv ]]; then
  python3 "$LH/perf/placement_check.py" nodes --candidates "$OUT/stage1.regressions.csv" \
    --map "$LH/perf_data/runs/s1h-$CASE/**/*.windowhash" --cap "$CAP" --out "$OUT/nodes.txt"
  T1=$(date +%s)
  for side in base head; do
    L=$LB; [[ $side == head ]] && L=$LH
    rm -rf "$L"/perf_data/runs/p{0,4,20,36,52}
    (cd "$L" && LLK_PERF_RUN_TYPES=$RUN_TYPES LLK_PERF_NODE_LIST="$OUT/nodes.txt" RUNNER_TEMP=$W/tmp/$CASE-s2-$side \
       ./tests/run_llk_perf_placement.sh > "$OUT/s2-$side.pytest.log" 2>&1)
    mkdir -p "$OUT/s2-$side"
    cp -r "$L"/perf_data/runs/p{0,4,20,36,52} "$OUT/s2-$side/" 2>/dev/null
  done
  echo "stage2_seconds=$(( $(date +%s) - T1 ))" >> "$OUT/timing"
  python3 "$LH/perf/placement_check.py" decide --threshold 0.02 --min-cycles 30 --module-threshold "" \
    --candidates "$OUT/stage1.regressions.csv" --baseline "$OUT/s2-base" --current "$OUT/s2-head" \
    --over-cap "$OUT/nodes.over" --arch "$ARCH" --report "$OUT/stage2.md"
fi
rm -rf "$W/tmp/$CASE-"*
echo "done" > "$OUT/status"
echo "=== $(date -u +%FT%TZ) case $CASE done"
