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

# EXPERIMENT (not for merge): 2x2 cell "no neighbours + serial" -- the noise floor.
# matmul only (no other perf modules on the board) AND a single xdist worker, so one
# Tensix at a time. Anything still moving here is neither neighbour contention nor
# concurrency.  PERF_ONLY="" restores the full test set.
PERF_ONLY="${PERF_ONLY--k perf_math_matmul}"
PYTEST_COMPILE_EXTRA="-q --override-ini=log_cli=false $PERF_ONLY"
PYTEST_RUN_EXTRA="-q --override-ini=log_cli=false $PERF_ONLY"

# With -k, a shard holding no matmul items collects nothing and pytest exits 5; that is
# an expected outcome here, so only exit code 5 is swallowed.
run_pytest() {
    local rc=0
    pytest "$@" || rc=$?
    if [ "$rc" -eq 5 ]; then
        echo "[experiment] no items collected for this shard; skipping it."
        exit 0
    fi
    return "$rc"
}

run_pytest $PYTEST_COMPILE_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" --compile-producer -n 10 -m "perf and not accuracy" --timeout=60 \
  --splits "$N_GROUPS" --group "$GROUP" \
  --junitxml="pytest-report-blackhole-${GROUP}-compile.xml" .
run_pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" --compile-consumer -n 1 -x -m "perf and not accuracy" --timeout=60 \
  --splits "$N_GROUPS" --group "$GROUP" \
  --junitxml="pytest-report-blackhole-${GROUP}-run.xml" .
junitparser merge pytest-report-blackhole-${GROUP}-compile.xml pytest-report-blackhole-${GROUP}-run.xml pytest-report-blackhole-${GROUP}.xml
