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

if [[ "$PHASE" != run ]]; then
  pytest $PYTEST_COMPILE_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" "${DEBUG_SYMBOL_ARGS[@]}" --compile-producer -n "$COMPILE_WORKERS" -m "perf and not accuracy" --timeout=60 \
    --perf-splits "$N_GROUPS" --perf-group "$GROUP" \
    --junitxml="$COMPILE_XML" .
fi
[[ "$PHASE" == compile ]] && exit 0

# EXPERIMENT (perf-trial/reset-recheck-*, not for merge): does `tt-smi -r all` between
# perf suites still move any config on post-#59447/#59486 main?
#
# The compile pass and the shard composition are unchanged: the measure pass first collects
# this shard's node ids with the same --perf-splits cut as stock, then runs them one test
# file (suite) per pytest invocation, in collection order, with the same --dist loadgroup
# -n 15 -x. PERF_RESET=1 issues `tt-smi -r all` before each suite; PERF_RESET=0 is the same
# per-suite split without the reset (the fair control for the reset).
PERF_RESET="${PERF_RESET:-1}"
echo "[reset-recheck] PERF_RESET=$PERF_RESET"
tt-smi -ls 2>&1 | sed 's/^/[reset-recheck]   /' || echo "[reset-recheck] WARNING: tt-smi -ls failed"

maybe_reset() {
  [[ "$PERF_RESET" == 1 ]] || return 0
  echo "[reset-recheck] $(date -u +%T) tt-smi -r all before $1"
  if ! tt-smi -r all; then
    echo "[reset-recheck] tt-smi -r all failed; trying tt-smi -r"
    tt-smi -r || { echo "[reset-recheck] ERROR: chip reset failed before $1" >&2; exit 3; }
  fi
}

NODE_DIR="$(mktemp -d)"
pytest -q --override-ini=log_cli=false --collect-only "${SPEED_OF_LIGHT_ARGS[@]}" "${DEBUG_SYMBOL_ARGS[@]}" --compile-consumer \
  -m "perf and not accuracy" --perf-splits "$N_GROUPS" --perf-group "$GROUP" . \
  | grep -E '^[^ ]+\.py::' > "$NODE_DIR/all.txt" || true
N_ALL=$(wc -l < "$NODE_DIR/all.txt" | tr -d ' ')
[[ "$N_ALL" -gt 0 ]] || { echo "[reset-recheck] ERROR: collected no node ids" >&2; exit 1; }
SUITES=$(sed 's/::.*//' "$NODE_DIR/all.txt" | awk '!seen[$0]++')
echo "[reset-recheck] $N_ALL tests in $(echo "$SUITES" | wc -l | tr -d ' ') suites:"
for pf in $SUITES; do
  echo "[reset-recheck]   $pf $(grep -c "^$pf::" "$NODE_DIR/all.txt")"
done

RUN_XMLS=()
for pf in $SUITES; do
  tag=$(basename "$pf" .py)
  grep "^$pf::" "$NODE_DIR/all.txt" > "$NODE_DIR/$tag.args"
  maybe_reset "$tag"
  xml="${REPORT}-${tag}-suite.xml"
  rc=0
  pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" "${DEBUG_SYMBOL_ARGS[@]}" --compile-consumer --dist loadgroup -n 15 "${EXITFIRST_ARGS[@]}" -m "perf and not accuracy" --timeout=60 \
    --junitxml="$xml" "@$NODE_DIR/$tag.args" || rc=$?
  [[ "$rc" -eq 5 ]] && rc=0
  [[ -f "$xml" ]] && RUN_XMLS+=("$xml")
  [[ "$rc" -eq 0 ]] || { echo "[reset-recheck] $tag exited $rc" >&2; exit "$rc"; }
done
junitparser merge "${RUN_XMLS[@]}" "$RUN_XML"
N_RAN=$(grep -o '<testcase ' "$RUN_XML" | wc -l | tr -d ' ')
echo "[reset-recheck] junit testcases $N_RAN of $N_ALL collected"
if [[ -f "$COMPILE_XML" ]]; then
  junitparser merge "$COMPILE_XML" "$RUN_XML" "${REPORT}.xml"
else
  cp "$RUN_XML" "${REPORT}.xml"
fi
