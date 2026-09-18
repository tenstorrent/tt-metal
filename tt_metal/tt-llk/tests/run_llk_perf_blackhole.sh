#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Blackhole LLK perf runner, shared by the 5 bh matrix groups in
# tests/pipeline_reorg/llk_perf_tests.yaml (the group index is passed in).
#
# pytest-split sharding: compile this shard's items (producer), then measure
# them (consumer) -- one invocation each over the whole perf suite.
#
# Usage: SPEED_OF_LIGHT=<true|false> run_llk_perf_blackhole.sh <group> <n_groups>
set -euo pipefail

GROUP="${1:?usage: run_llk_perf_blackhole.sh <group> <n_groups>}"
N_GROUPS="${2:?usage: run_llk_perf_blackhole.sh <group> <n_groups>}"
SPEED_OF_LIGHT="${SPEED_OF_LIGHT:-true}"

case "$SPEED_OF_LIGHT" in
  true)
    SPEED_OF_LIGHT_ARGS=(--speed-of-light)
    ;;
  false)
    SPEED_OF_LIGHT_ARGS=()
    ;;
  *)
    echo "SPEED_OF_LIGHT must be 'true' or 'false', got '$SPEED_OF_LIGHT'" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/python_tests"
mkdir -p perf_data

PYTEST_COMPILE_EXTRA="-q --override-ini=log_cli=false"
PYTEST_RUN_EXTRA="-q --override-ini=log_cli=false"

# EXPERIMENT (not for merge): reproduce the execution shape of #46478 -- one pytest
# invocation PER TEST FILE, so each suite's variants still fan out over the xdist
# workers, but suites run one after another on the same board. Cross-suite state
# carry-over is therefore present, which the matmul-only runs could not test
# (matmul was the only suite on the board there).
#
# Distribution: stock pytest-split, applied PER FILE. Each file's items are cut
# into $N_GROUPS equal contiguous chunks and this shard takes chunk $GROUP, so the
# shards cover every file exactly once between them -- complete, non-overlapping,
# and with no custom packer. That is the difference from #46478, whose
# perf_bin_pack.py assigned whole slices and dropped 6,640 matmul configs.
#
# Files are passed as explicit pytest targets rather than via -k: several suite
# names are substrings of others (perf_pack / perf_pack_untilize,
# perf_fast_untilize / perf_fast_untilize_baseline_compare, perf_eltwise_binary /
# perf_eltwise_binary_sfpu), so -k would merge suites and run the longer-named
# ones twice.
#
# PERF_RESET=1 issues `tt-smi -r 0` between suites: the per-slice chip reset that
# #46478 added and #51157 removed.
PERF_RESET="${PERF_RESET:-0}"

maybe_reset() {
    [ "$PERF_RESET" = "1" ] || return 0
    echo "[experiment] tt-smi -r 0 before $1"
    tt-smi -r 0 || echo "[experiment] WARNING: tt-smi reset failed; continuing"
}

# Enumerate perf test FILES from the full collection (not this shard's slice), so
# every shard iterates the same list.
PERF_FILES=$(pytest -q --collect-only -m "perf and not accuracy" . 2>/dev/null \
  | grep '::' | sed 's/::.*//' | sort -u)
if [ -z "$PERF_FILES" ]; then
    echo "[experiment] ERROR: collected no perf test files" >&2
    exit 1
fi
echo "[experiment] suites ($(echo "$PERF_FILES" | wc -l | tr -d ' ')):"
echo "$PERF_FILES" | sed 's/^/[experiment]   /'

# A file with no items in this shard's chunk exits 5; expected, not a failure.
run_pytest() {
    local rc=0
    pytest "$@" || rc=$?
    [ "$rc" -eq 5 ] && return 0
    return "$rc"
}

for pf in $PERF_FILES; do
    tag=$(basename "$pf" .py)
    maybe_reset "$tag (compile)"
    run_pytest $PYTEST_COMPILE_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" --compile-producer -n 10 \
      -m "perf and not accuracy" --timeout=60 \
      --splits "$N_GROUPS" --group "$GROUP" \
      --junitxml="pytest-report-blackhole-${GROUP}-${tag}-compile.xml" "$pf"
    maybe_reset "$tag (measure)"
    run_pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" --compile-consumer -n 15 -x \
      -m "perf and not accuracy" --timeout=60 \
      --splits "$N_GROUPS" --group "$GROUP" \
      --junitxml="pytest-report-blackhole-${GROUP}-${tag}-run.xml" "$pf"
done

junitparser merge pytest-report-blackhole-${GROUP}-*-compile.xml pytest-report-blackhole-${GROUP}-*-run.xml pytest-report-blackhole-${GROUP}.xml
