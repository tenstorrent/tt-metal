#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Blackhole LLK perf runner, shared by the 5 bh matrix groups in
# tests/pipeline_reorg/llk_perf_tests.yaml (the group index is passed in).
#
# Per slice: compile only this slice's items (producer), reset the board,
# then measure (consumer). Slicing the producer keeps each item compiled
# exactly once across the run (like main's --splits/--group) instead of
# recompiling whole giant files in every group holding one of their slices
# -- the over-compile that was timing the producer out before the consumer
# ever ran. The per-slice `tt-smi -r 0` is the reset-between-tests fix this
# branch exists for.
#
# Usage: run_llk_perf_blackhole.sh <group> <n_groups>
set -euo pipefail

GROUP="${1:?usage: run_llk_perf_blackhole.sh <group> <n_groups>}"
N_GROUPS="${2:?usage: run_llk_perf_blackhole.sh <group> <n_groups>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/python_tests"
mkdir -p perf_data

PYTEST_COMPILE_EXTRA="-q --override-ini=log_cli=false"
PYTEST_RUN_EXTRA="-q --override-ini=log_cli=false"

# Slice-level greedy bin-pack with an adaptive target (see perf_bin_pack.py).
# Emits this group's slices as file:group_idx:total_splits:items.
ASSIGNMENTS=$(python3 "$SCRIPT_DIR/perf_bin_pack.py" "$GROUP" "$N_GROUPS")
echo "Group ${GROUP}/${N_GROUPS} slices:"
echo "$ASSIGNMENTS"

while IFS=":" read -r FILE GIDX TOTAL ITEMS; do
  [ -z "$FILE" ] && continue
  tname="${FILE%.py}"

  if [ "$TOTAL" = "1" ]; then
    SPLIT_ARGS=""
    REPORT_NAME="${tname}-run"
  else
    SPLIT_ARGS="--splits ${TOTAL} --group ${GIDX}"
    REPORT_NAME="${tname}-s${GIDX}of${TOTAL}-run"
  fi

  pytest $PYTEST_COMPILE_EXTRA --speed-of-light --compile-producer -n 10 \
    -m "perf and not accuracy" --timeout=60 \
    $SPLIT_ARGS \
    "$FILE"
  tt-smi -r 0
  pytest $PYTEST_RUN_EXTRA --speed-of-light --compile-consumer -n 15 -x \
    -m "perf and not accuracy" --timeout=60 \
    $SPLIT_ARGS \
    --junitxml="pytest-report-blackhole-${GROUP}-${REPORT_NAME}.xml" \
    "$FILE"
done <<< "$ASSIGNMENTS"

# Merge this group's per-slice run reports. Guard the glob with nullglob so an
# early exit that produced no per-slice reports doesn't turn a real failure
# into a confusing junitparser "file not found".
shopt -s nullglob
run_reports=(pytest-report-blackhole-${GROUP}-*-run.xml)
shopt -u nullglob
if [ "${#run_reports[@]}" -gt 0 ]; then
  junitparser merge "${run_reports[@]}" pytest-report-blackhole-${GROUP}.xml
else
  echo "No per-slice run reports for group ${GROUP}; skipping merge."
fi
