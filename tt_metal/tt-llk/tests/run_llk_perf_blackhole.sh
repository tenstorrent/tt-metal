#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Blackhole LLK perf runner, shared by the 5 bh matrix groups in
# tests/pipeline_reorg/llk_perf_tests.yaml (the group index is passed in).
#
# Usage: SPEED_OF_LIGHT=<true|false> run_llk_perf_blackhole.sh <group> <n_groups>
#
# Optional, for a compile pass that runs on another (CPU-only) runner:
#   LLK_PERF_PHASE=all|compile|run  all (default) compiles, then measures;
#                                   compile only builds the ELFs; run only measures
#                                   ELFs that are already in the build directory.
#   LLK_PERF_COMPILE_WORKERS=N      producer workers (default 10).
#   LLK_PERF_DEBUG_SYMBOLS=true|false  false builds without -g (default true).
#   LLK_PERF_EXITFIRST=true|false   false runs the measuring pass without -x (default true).
set -euo pipefail

GROUP="${1:?usage: run_llk_perf_blackhole.sh <group> <n_groups>}"
N_GROUPS="${2:?usage: run_llk_perf_blackhole.sh <group> <n_groups>}"
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

PHASE="${LLK_PERF_PHASE:-all}"
case "$PHASE" in
  all|compile|run) ;;
  *)
    echo "LLK_PERF_PHASE must be 'all', 'compile' or 'run', got '$PHASE'" >&2
    exit 2
    ;;
esac
COMPILE_WORKERS="${LLK_PERF_COMPILE_WORKERS:-10}"

# Both passes take the flag: the consumer does not compile, and the variant id
# ignores compile flags, so the two must agree for the ELFs to be found.
case "${LLK_PERF_DEBUG_SYMBOLS:-true}" in
  true) DEBUG_SYMBOL_ARGS=() ;;
  false) DEBUG_SYMBOL_ARGS=(--no-debug-symbols) ;;
  *)
    echo "LLK_PERF_DEBUG_SYMBOLS must be 'true' or 'false', got '$LLK_PERF_DEBUG_SYMBOLS'" >&2
    exit 2
    ;;
esac

case "${LLK_PERF_EXITFIRST:-true}" in
  true) EXITFIRST_ARGS=(-x) ;;
  false) EXITFIRST_ARGS=() ;;
  *)
    echo "LLK_PERF_EXITFIRST must be 'true' or 'false', got '$LLK_PERF_EXITFIRST'" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/python_tests"
mkdir -p perf_data

PYTEST_COMPILE_EXTRA="-q --override-ini=log_cli=false"
PYTEST_RUN_EXTRA="-q --override-ini=log_cli=false"

COMPILE_XML="pytest-report-blackhole-${GROUP}-compile.xml"
RUN_XML="pytest-report-blackhole-${GROUP}-run.xml"

if [[ "$PHASE" != run ]]; then
  pytest $PYTEST_COMPILE_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" "${DEBUG_SYMBOL_ARGS[@]}" --compile-producer -n "$COMPILE_WORKERS" -m "perf and not accuracy" --timeout=60 \
    --perf-splits "$N_GROUPS" --perf-group "$GROUP" \
    --junitxml="$COMPILE_XML" .
fi
[[ "$PHASE" == compile ]] && exit 0

pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" "${DEBUG_SYMBOL_ARGS[@]}" --compile-consumer --dist loadgroup -n 15 "${EXITFIRST_ARGS[@]}" -m "perf and not accuracy" --timeout=60 \
  --perf-splits "$N_GROUPS" --perf-group "$GROUP" \
  --junitxml="$RUN_XML" .
# A run-only phase merges the compile report that came with the ELFs, when there is one.
if [[ -f "$COMPILE_XML" ]]; then
  junitparser merge "$COMPILE_XML" "$RUN_XML" pytest-report-blackhole-${GROUP}.xml
else
  cp "$RUN_XML" pytest-report-blackhole-${GROUP}.xml
fi
