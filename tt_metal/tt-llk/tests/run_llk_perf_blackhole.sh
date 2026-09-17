#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Blackhole LLK perf runner. The matrix in
# tests/pipeline_reorg/llk_perf_tests.yaml passes a split_group (for JUnit /
# artefact names) and the pytest selector for that shard.
#
# Compile this shard's items (producer), then measure them (consumer).
# Both invocations must receive the same selector.
#
# Usage: SPEED_OF_LIGHT=<true|false> run_llk_perf_blackhole.sh <group> <pytest args...>
set -euo pipefail

GROUP="${1:?usage: run_llk_perf_blackhole.sh <group> <pytest args...>}"
shift
if [[ $# -lt 1 ]]; then
  echo "usage: run_llk_perf_blackhole.sh <group> <pytest args...>" >&2
  exit 2
fi
SPEED_OF_LIGHT="${SPEED_OF_LIGHT:-true}"
export TT_LLK_DISABLE_ASSERTS="${TT_LLK_DISABLE_ASSERTS:-1}"

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

pytest $PYTEST_COMPILE_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" --compile-producer -n 10 -m "perf and not accuracy" --timeout=60 \
  --junitxml="pytest-report-blackhole-${GROUP}-compile.xml" "$@"
pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" --compile-consumer -n 15 -x -m "perf and not accuracy" --timeout=60 \
  --junitxml="pytest-report-blackhole-${GROUP}-run.xml" "$@"
junitparser merge pytest-report-blackhole-${GROUP}-compile.xml pytest-report-blackhole-${GROUP}-run.xml pytest-report-blackhole-${GROUP}.xml
