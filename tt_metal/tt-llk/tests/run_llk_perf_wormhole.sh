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
#
# Verification matrix knobs (transient CI use only, not for production):
#   PERF_NC_WC=nc|wc       — nc: no counters (default); wc: add --enable-perf-counters
#   PERF_SOL=sol|nosol     — sol: speed-of-light on (default); nosol: off
#   PERF_ORDER=fwd|rev     — fwd: forward collection order (default); rev: reversed
# When PERF_NC_WC/PERF_SOL/PERF_ORDER are set, the script runs a single ordered consumer
# pass (1 worker, @ids file) instead of the normal 15-worker dist-loadgroup pass.
set -euo pipefail

GROUP="${1:?usage: run_llk_perf_wormhole.sh <group> <n_groups>}"
N_GROUPS="${2:?usage: run_llk_perf_wormhole.sh <group> <n_groups>}"

# --- verification matrix knobs -------------------------------------------
PERF_NC_WC="${PERF_NC_WC:-}"
PERF_SOL="${PERF_SOL:-}"
PERF_ORDER="${PERF_ORDER:-}"

# When any verification knob is set, derive SPEED_OF_LIGHT from PERF_SOL.
if [[ -n "$PERF_NC_WC" || -n "$PERF_SOL" || -n "$PERF_ORDER" ]]; then
  VERIFY_MODE=1
  case "${PERF_NC_WC:-nc}" in
    nc) PERF_COUNTER_ARGS=() ;;
    wc) PERF_COUNTER_ARGS=(--enable-perf-counters) ;;
    *) echo "PERF_NC_WC must be 'nc' or 'wc', got '${PERF_NC_WC}'" >&2; exit 2 ;;
  esac
  case "${PERF_SOL:-sol}" in
    sol) SPEED_OF_LIGHT=true ;;
    nosol) SPEED_OF_LIGHT=false ;;
    *) echo "PERF_SOL must be 'sol' or 'nosol', got '${PERF_SOL}'" >&2; exit 2 ;;
  esac
  PERF_ORDER="${PERF_ORDER:-fwd}"
  case "$PERF_ORDER" in
    fwd|rev) ;;
    *) echo "PERF_ORDER must be 'fwd' or 'rev', got '${PERF_ORDER}'" >&2; exit 2 ;;
  esac
  export PERF_KEEP_RUNS=100
else
  VERIFY_MODE=0
  PERF_COUNTER_ARGS=()
  SPEED_OF_LIGHT="${SPEED_OF_LIGHT:-true}"
fi
# -------------------------------------------------------------------------

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

if [[ "$PHASE" != run ]]; then
  pytest $PYTEST_COMPILE_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" "${DEBUG_SYMBOL_ARGS[@]}" --compile-producer -n "$COMPILE_WORKERS" -m "perf and not accuracy" --timeout=60 \
    --perf-splits "$N_GROUPS" --perf-group "$GROUP" \
    --junitxml="$COMPILE_XML" .
fi
[[ "$PHASE" == compile ]] && exit 0

if [[ "$VERIFY_MODE" == 1 ]]; then
  # Collect test IDs for this shard in collection order, then optionally reverse.
  pytest -q --collect-only --compile-consumer "${SPEED_OF_LIGHT_ARGS[@]}" "${PERF_COUNTER_ARGS[@]}" \
    -p no:cacheprovider -m "perf and not accuracy" \
    --perf-splits "$N_GROUPS" --perf-group "$GROUP" . 2>/dev/null | grep '::' > grp_ids.txt || true
  tac grp_ids.txt > grp_ids_rev.txt
  wc -l grp_ids.txt >&2
  if [[ "$PERF_ORDER" == rev ]]; then
    ID_FILE=grp_ids_rev.txt
  else
    ID_FILE=grp_ids.txt
  fi
  TAG="${PERF_NC_WC:-nc}_${PERF_SOL:-sol}_${PERF_ORDER:-fwd}"
  PERF_RUN_TAG="$TAG" pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" "${DEBUG_SYMBOL_ARGS[@]}" "${PERF_COUNTER_ARGS[@]}" \
    --compile-consumer -n 1 -p no:cacheprovider --timeout=600 "@$ID_FILE" \
    --junitxml="$RUN_XML" || echo "verify consumer rc=$? (non-fatal)"
  cp "$COMPILE_XML" "${REPORT}.xml" 2>/dev/null || cp "$RUN_XML" "${REPORT}.xml" 2>/dev/null || true
  exit 0
fi

pytest $PYTEST_RUN_EXTRA "${SPEED_OF_LIGHT_ARGS[@]}" "${DEBUG_SYMBOL_ARGS[@]}" --compile-consumer --dist loadgroup -n 15 "${EXITFIRST_ARGS[@]}" -m "perf and not accuracy" --timeout=60 \
  --perf-splits "$N_GROUPS" --perf-group "$GROUP" \
  --junitxml="$RUN_XML" .
if [[ -f "$COMPILE_XML" ]]; then
  junitparser merge "$COMPILE_XML" "$RUN_XML" "${REPORT}.xml"
else
  cp "$RUN_XML" "${REPORT}.xml"
fi
