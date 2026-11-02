#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Blackhole LLK perf runner, shared by the 5 bh matrix groups in
# tests/pipeline_reorg/llk_perf_tests.yaml (the group index is passed in).
#
# Usage: SPEED_OF_LIGHT=<true|false> run_llk_perf_blackhole.sh <group> <n_groups>
# Optional: LLK_PERF_PHASE=all|compile|run, LLK_PERF_COMPILE_WORKERS, LLK_PERF_DEBUG_SYMBOLS,
# LLK_PERF_EXITFIRST, LLK_PERF_REPORT_TAG (JUnit name suffix). Defaults keep today's commands.
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

REPORT="pytest-report-blackhole-${GROUP}${LLK_PERF_REPORT_TAG:-}"
COMPILE_XML="${REPORT}-compile.xml"
RUN_XML="${REPORT}-run.xml"

# Experiment: only group 1 runs a small sample (WH outlier set + every 100th BH test), single worker, with and
# without the warm-up on the same card.
if [[ "$GROUP" != 1 ]]; then
  echo "experiment: only group 1 runs"
  exit 0
fi
export PERF_KEEP_RUNS=100
pytest -q --collect-only --compile-producer "${SPEED_OF_LIGHT_ARGS[@]}" -p no:cacheprovider -m "perf and not accuracy and not quasar" perf_*.py 2> /dev/null | grep '::' > all_ids.txt || true
{ grep -F -x -f exp_bh_ids.txt all_ids.txt || true; awk 'NR % 100 == 1' all_ids.txt; } | awk '!seen[$0]++' > exp_ids.txt
tac exp_ids.txt > exp_ids_rev.txt
wc -l all_ids.txt exp_ids.txt
pytest $PYTEST_COMPILE_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" "${DEBUG_SYMBOL_ARGS[@]}" --compile-producer -n "$COMPILE_WORKERS" --timeout=600 \
  --junitxml="$COMPILE_XML" @exp_ids.txt
run() {
  local tag=$1 ids=$2
  shift 2
  env PERF_RUN_TAG="$tag" "$@" pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" "${DEBUG_SYMBOL_ARGS[@]}" --compile-consumer -n 1 \
    --timeout=600 -p no:cacheprovider @"$ids" || echo "run $tag rc=$?"
  echo "done $tag"
}
run nowarm_seq1 exp_ids.txt LLK_PERF_NO_WARMUP=1
run nowarm_seq2 exp_ids.txt LLK_PERF_NO_WARMUP=1
run nowarm_rev exp_ids_rev.txt LLK_PERF_NO_WARMUP=1
run nowarm_l1_seq exp_ids.txt LLK_PERF_NO_WARMUP=1 LLK_PERF_RUN_TYPES=L1_TO_L1
run nowarm_l1_rev exp_ids_rev.txt LLK_PERF_NO_WARMUP=1 LLK_PERF_RUN_TYPES=L1_TO_L1
run warm_seq1 exp_ids.txt
run warm_seq2 exp_ids.txt
run warm_rev exp_ids_rev.txt
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
