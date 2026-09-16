#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Blackhole LLK perf runner, shared by the 5 bh matrix groups in
# tests/pipeline_reorg/llk_perf_tests.yaml (the group index is passed in).
#
# Each perf module is sharded independently. For every non-empty module shard,
# compile its items (producer), reset the board, then measure them in a fresh
# consumer process. This preserves full pytest-split coverage without the
# retired slice bin-pack, which could assign multiple slices of one module to
# the same CI shard and overwrite part of that module's performance output.
#
# Usage: SPEED_OF_LIGHT=<true|false> LLK_DISABLE_PERF_RELEVANCE=<0|1> \
#        TT_LLK_DISABLE_ASSERTS=<0|1> run_llk_perf_blackhole.sh <group> <n_groups>
#
# LLK_DISABLE_PERF_RELEVANCE=1 skips isolate reuse (A/B). Default 0.
# TT_LLK_DISABLE_ASSERTS=1 (default) compiles without LLK_ASSERT/ebreak.
set -euo pipefail

GROUP="${1:?usage: run_llk_perf_blackhole.sh <group> <n_groups>}"
N_GROUPS="${2:?usage: run_llk_perf_blackhole.sh <group> <n_groups>}"
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

case "$TT_LLK_DISABLE_ASSERTS" in
  0|1) ;;
  *)
    echo "TT_LLK_DISABLE_ASSERTS must be '0' or '1', got '$TT_LLK_DISABLE_ASSERTS'" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/python_tests"
mkdir -p perf_data

PYTEST_COMPILE_EXTRA="-q --override-ini=log_cli=false"
PYTEST_RUN_EXTRA="-q --override-ini=log_cli=false"

reports=()
for file in perf_*.py; do
  test_name="${file%.py}"
  compile_report="pytest-report-blackhole-${GROUP}-${test_name}-compile.xml"
  run_report="pytest-report-blackhole-${GROUP}-${test_name}-run.xml"

  echo "Compiling ${file}, group ${GROUP}/${N_GROUPS}"
  if pytest $PYTEST_COMPILE_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" --compile-producer -n 10 \
    -m "perf and not accuracy" --timeout=60 \
    --splits "$N_GROUPS" --group "$GROUP" \
    --junitxml="$compile_report" "$file"; then
    reports+=("$compile_report")
  else
    status=$?
    if [ "$status" -eq 5 ]; then
      echo "No selected tests in ${file}, group ${GROUP}/${N_GROUPS}; skipping"
      rm -f "$compile_report"
      continue
    fi
    exit "$status"
  fi

  echo "Resetting the board before measuring ${file}, group ${GROUP}/${N_GROUPS}"
  tt-smi -r 0

  pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" --compile-consumer -n 15 -x \
    -m "perf and not accuracy" --timeout=60 \
    --splits "$N_GROUPS" --group "$GROUP" \
    --junitxml="$run_report" "$file"
  reports+=("$run_report")
done

if [ "${#reports[@]}" -eq 0 ]; then
  echo "No Blackhole perf tests selected for group ${GROUP}/${N_GROUPS}" >&2
  exit 5
fi

junitparser merge "${reports[@]}" "pytest-report-blackhole-${GROUP}.xml"
