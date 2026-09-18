#!/usr/bin/env bash
# Usage: run_profile.sh <config.yaml> <mgd.textproto> <name> [train.py args...]
# Runs train.py under the tt-metal Tracy device profiler; copies ops_perf_results CSV to results/<name>_ops.csv
set -euo pipefail
ulimit -u 65536 2>/dev/null || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_TRAIN="$(cd "$HERE/../../.." && pwd)"
TT_METAL="$(cd "$TT_TRAIN/.." && pwd)"
CFG="$1"; MGD="$2"; NAME="$3"; shift 3
source "$TT_METAL/python_env/bin/activate"
export TT_METAL_HOME="$TT_METAL" TT_METAL_RUNTIME_ROOT="$TT_METAL" TT_MESH_GRAPH_DESC_PATH="$MGD"
export TT_LOGGER_LEVEL="${TT_LOGGER_LEVEL:-error}"
unset TTML_NAIVE_PROFILER
cd "$TT_TRAIN"
LOG="$HERE/results/$NAME.profile.log"
echo "== $(date -Is) profiling cfg=$CFG -> $LOG"
env -u TT_METAL_DPRINT_CORES TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000 \
  python -m tracy -r -v -p --no-op-info-cache -n "$NAME" sources/examples/train/train.py -c "$CFG" "$@" > "$LOG" 2>&1 || { echo "FAILED"; tail -30 "$LOG"; exit 1; }
CSV=$(ls -t "$TT_METAL"/generated/profiler/reports/*/ops_perf_results_*.csv "$TT_METAL"/generated/profiler/reports/*/*/ops_perf_results_*.csv | head -1)
cp "$CSV" "$HERE/results/${NAME}_ops.csv"
echo "ops csv -> $HERE/results/${NAME}_ops.csv ($(wc -l < "$CSV") rows)"
