#!/usr/bin/env bash
# Usage: run_one.sh <config.yaml> <mgd.textproto> <logname> [extra train.py args...]
# Runs train.py with the naive phase profiler enabled and stores the log under results/.
set -euo pipefail
ulimit -u 65536 2>/dev/null || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_TRAIN="$(cd "$HERE/../../.." && pwd)"
TT_METAL="$(cd "$TT_TRAIN/.." && pwd)"
CFG="$1"; MGD="$2"; NAME="$3"; shift 3
source "$TT_METAL/python_env/bin/activate"
export TT_METAL_HOME="$TT_METAL"
export TT_METAL_RUNTIME_ROOT="$TT_METAL"
export TT_MESH_GRAPH_DESC_PATH="$MGD"
export TT_LOGGER_LEVEL="${TT_LOGGER_LEVEL:-error}"
export TTML_NAIVE_PROFILER="${TTML_NAIVE_PROFILER:-1}"
cd "$TT_TRAIN"
LOG="$HERE/results/$NAME.log"
echo "== $(date -Is) cfg=$CFG mgd=$MGD args=$* -> $LOG"
/usr/bin/time -v python sources/examples/train/train.py -c "$CFG" "$@" > "$LOG" 2>&1 || { echo "FAILED rc=$?"; tail -40 "$LOG"; exit 1; }
grep -E "^Step:|Total time|MFU|TPS" "$LOG" | tail -15
