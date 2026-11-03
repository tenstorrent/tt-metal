#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Wormhole LLK perf runner, shared by the 5 wh matrix groups in
# tests/pipeline_reorg/llk_perf_tests.yaml (the group index is passed in).
#
# Usage: SPEED_OF_LIGHT=<true|false> run_llk_perf_wormhole.sh <group> <n_groups>
# Optional: LLK_PERF_PHASE=all|compile|run, LLK_PERF_COMPILE_WORKERS, LLK_PERF_DEBUG_SYMBOLS,
# LLK_PERF_EXITFIRST, LLK_PERF_REPORT_TAG (JUnit name suffix). Defaults keep today's commands.
set -euo pipefail

GROUP="${1:?usage: run_llk_perf_wormhole.sh <group> <n_groups>}"
N_GROUPS="${2:?usage: run_llk_perf_wormhole.sh <group> <n_groups>}"
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

REPORT="pytest-report-wormhole-${GROUP}${LLK_PERF_REPORT_TAG:-}"
COMPILE_XML="${REPORT}-compile.xml"
RUN_XML="${REPORT}-run.xml"

export RUNS_SEQ=0
export RUN_REV=0
export RUNS_X=2
# Experiment: problem 1 check of the stack (1/5 to 3/5). One card runs this group single-worker in collection order
# RUNS_SEQ times, then reversed when RUN_REV=1, then CI style (15 workers) RUNS_X times; each run keeps its own perf_data.
export PERF_KEEP_RUNS=100
pytest $PYTEST_COMPILE_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" "${DEBUG_SYMBOL_ARGS[@]}" --compile-producer -n "$COMPILE_WORKERS" -m "perf and not accuracy" --timeout=600 \
  --perf-splits "$N_GROUPS" --perf-group "$GROUP" --junitxml="$COMPILE_XML" .
pytest -q --collect-only --compile-consumer "${SPEED_OF_LIGHT_ARGS[@]}" -p no:cacheprovider -m "perf and not accuracy" \
  --perf-splits "$N_GROUPS" --perf-group "$GROUP" . 2> /dev/null | grep '::' > grp_ids.txt || true
tac grp_ids.txt > grp_ids_rev.txt
wc -l grp_ids.txt
run() {
  local tag=$1
  shift
  env PERF_RUN_TAG="$tag" pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" "${DEBUG_SYMBOL_ARGS[@]}" --compile-consumer --timeout=600 \
    -p no:cacheprovider "$@" || echo "run $tag rc=$?"
  echo "done $tag"
}
for i in $(seq 1 "${RUNS_SEQ:-2}"); do run "seq$i" -n 1 @grp_ids.txt; done
[[ "${RUN_REV:-0}" == 1 ]] && run rev -n 1 @grp_ids_rev.txt
for i in $(seq 1 "${RUNS_X:-0}"); do run "x$i" --dist loadgroup -n 15 -m "perf and not accuracy" --perf-splits "$N_GROUPS" --perf-group "$GROUP" .; done
cp "$COMPILE_XML" "${REPORT}.xml"
exit 0

if [[ "$PHASE" != run ]]; then
  pytest $PYTEST_COMPILE_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" "${DEBUG_SYMBOL_ARGS[@]}" --compile-producer -n "$COMPILE_WORKERS" -m "perf and not accuracy" --timeout=60 \
    --perf-splits "$N_GROUPS" --perf-group "$GROUP" \
    --junitxml="$COMPILE_XML" .
fi
[[ "$PHASE" == compile ]] && exit 0

pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" "${DEBUG_SYMBOL_ARGS[@]}" --compile-consumer --dist loadgroup -n 15 "${EXITFIRST_ARGS[@]}" -m "perf and not accuracy" --timeout=60 \
  --perf-splits "$N_GROUPS" --perf-group "$GROUP" \
  --junitxml="$RUN_XML" .
if [[ -f "$COMPILE_XML" ]]; then
  junitparser merge "$COMPILE_XML" "$RUN_XML" "${REPORT}.xml"
else
  cp "$RUN_XML" "${REPORT}.xml"
fi
