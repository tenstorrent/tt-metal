#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Wormhole LLK perf runner, shared by the 5 wh matrix groups in
# tests/pipeline_reorg/llk_perf_tests.yaml (the group index is passed in).
#
# pytest-split sharding: compile this shard's items (producer), then measure
# them (consumer) -- one invocation each over the whole perf suite.
#
# Usage: SPEED_OF_LIGHT=<true|false> LLK_DISABLE_PERF_RELEVANCE=<0|1> \
#        run_llk_perf_wormhole.sh <group> <n_groups>
#
# LLK_DISABLE_PERF_RELEVANCE=1 skips isolate reuse. Default 0.
set -euo pipefail

GROUP="${1:?usage: run_llk_perf_wormhole.sh <group> <n_groups>}"
N_GROUPS="${2:?usage: run_llk_perf_wormhole.sh <group> <n_groups>}"
SPEED_OF_LIGHT="${SPEED_OF_LIGHT:-true}"
export TT_LLK_DISABLE_ASSERTS="${TT_LLK_DISABLE_ASSERTS:-1}"
export LLK_DISABLE_PERF_RELEVANCE="${LLK_DISABLE_PERF_RELEVANCE:-0}"

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

case "$LLK_DISABLE_PERF_RELEVANCE" in
  0|1) ;;
  *)
    echo "LLK_DISABLE_PERF_RELEVANCE must be '0' or '1', got '$LLK_DISABLE_PERF_RELEVANCE'" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/python_tests"
mkdir -p perf_data

PYTEST_COMPILE_EXTRA="-q --override-ini=log_cli=false"
PYTEST_RUN_EXTRA="-q --override-ini=log_cli=false"

pytest $PYTEST_COMPILE_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" --compile-producer -n 10 -m "perf and not accuracy" --timeout=60 \
  --splits "$N_GROUPS" --group "$GROUP" \
  --junitxml="pytest-report-wormhole-${GROUP}-compile.xml" .
pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" --compile-consumer --dist loadgroup -n 15 -x -m "perf and not accuracy" --timeout=60 \
  --splits "$N_GROUPS" --group "$GROUP" \
  --junitxml="pytest-report-wormhole-${GROUP}-run.xml" .
junitparser merge pytest-report-wormhole-${GROUP}-compile.xml pytest-report-wormhole-${GROUP}-run.xml pytest-report-wormhole-${GROUP}.xml
