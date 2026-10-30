#!/bin/bash
# SEPARATE pre-pass: sweep matmul fidelity x dtype per shape and write a best-config warm-start table,
# to be run BEFORE the optimize loop. This does NOT invoke or modify the optimize tool (perf-mcp / the
# CC loop) at all -- it is a standalone step whose matmul_sweep.json you can inspect / seed the run with.
#
# Usage:
#   run_matmul_sweep.sh <perf_test_node> [k_case]
# Example:
#   run_matmul_sweep.sh "models/demos/hf_seamless_m4t_medium/tests/test_perf.py::test_perf" ""
#
# Optional env: PERF_MCP_MATMUL_SWEEP_OUT (output path), PERF_MCP_MATMUL_SWEEP_PCC (min pcc, def 0.99),
#   PERF_MCP_MATMUL_SWEEP_ITERS (timed reps, def 5), PERF_MCP_MATMUL_SWEEP_MAX_SHAPES (cap, def 0=all),
#   PERF_MCP_DEVICES (device ids).
set -euo pipefail

NODE="${1:?usage: run_matmul_sweep.sh <perf_test_node> [k_case]}"
CASE="${2:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

ARGS=("$NODE")
[ -n "$CASE" ] && ARGS+=(--case "$CASE")
[ -n "${PERF_MCP_MATMUL_SWEEP_OUT:-}" ] && ARGS+=(--out "$PERF_MCP_MATMUL_SWEEP_OUT")
[ -n "${PERF_MCP_MATMUL_SWEEP_PCC:-}" ] && ARGS+=(--pcc "$PERF_MCP_MATMUL_SWEEP_PCC")
[ -n "${PERF_MCP_MATMUL_SWEEP_ITERS:-}" ] && ARGS+=(--iters "$PERF_MCP_MATMUL_SWEEP_ITERS")
[ -n "${PERF_MCP_MATMUL_SWEEP_MAX_SHAPES:-}" ] && ARGS+=(--max-shapes "$PERF_MCP_MATMUL_SWEEP_MAX_SHAPES")

echo "[matmul-sweep] pre-pass on node: $NODE ${CASE:+(-k $CASE)}"
PYTHONPATH="$REPO_ROOT:$REPO_ROOT/models/experimental/perf_automation" \
  python -m cc_optimize.matmul_sweep "${ARGS[@]}"
echo "[matmul-sweep] done -- seed the optimize run from the matmul_sweep.json above."
