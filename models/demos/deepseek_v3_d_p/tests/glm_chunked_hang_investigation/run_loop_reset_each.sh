#!/bin/bash
# Same as run_loop.sh, but unconditionally resets the galaxy (tt-smi -glx_reset)
# before EVERY iteration, mirroring CI's fresh allocation / "Reset cards and
# validate cluster" step before each job run. Useful if you suspect residual
# state between iterations is masking the hang (it wasn't, in ~30 tries on a
# lower-power box -- see FINDINGS.md -- but worth trying on real hardware).
#
# Usage: ./run_loop_reset_each.sh [START] [END]        (default 16 30 -- picks
# up numbering where run_loop.sh's default 1..15 left off so logs coexist in
# the same LOGDIR; pass your own START if you ran run_loop.sh with different
# bounds).
# Env:   GLM_HANG_LOGDIR to override the default log directory.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/python_env/bin/activate" ]; then
  source "$REPO_ROOT/python_env/bin/activate"
else
  echo "note: $REPO_ROOT/python_env not found; assuming ttnn/pytest are already available in the current environment" >&2
fi

export MESH_DEVICE=TG
export LOGURU_LEVEL=INFO
export GLM52_HF_MODEL="${GLM52_HF_MODEL:-/mnt/models/deepseek-prefill-cache/GLM-5.2-FP8}"
export TT_GLM52_PREFILL_TTNN_CACHE="${TT_GLM52_PREFILL_TTNN_CACHE:-/mnt/models/deepseek-prefill-cache/glm52_ttnn_cache}"
export PREFILL_TRACE_DIR="${PREFILL_TRACE_DIR:-/mnt/models/deepseek-prefill-cache/glm-traces/vllm-glm52-indexer-kcache-55k}"

TEST='models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py::test_glm_prefill_transformer_chunked_no_pcc[blackhole-glm52-mesh-8x4-L78-preload0-chunks_eleven-ten_iters]'

LOGDIR="${GLM_HANG_LOGDIR:-$REPO_ROOT/generated/glm_chunked_hang_repro}"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.log"
CURRENT="$LOGDIR/current_run.info"
START="${1:-16}"
END="${2:-30}"

echo $$ > "$LOGDIR/runner.pid"
echo "reset-before-every-iteration loop started $(date -u -Iseconds), iterations $START..$END" | tee -a "$SUMMARY"

for i in $(seq "$START" "$END"); do
  LOG="$LOGDIR/run_${i}.log"

  echo "resetting before iteration $i: tt-smi -glx_reset" | tee -a "$SUMMARY"
  tt-smi -glx_reset > "$LOGDIR/glx_reset_before_${i}.log" 2>&1

  echo "=== iteration $i starting $(date -u -Iseconds) ===" | tee -a "$SUMMARY"

  mpirun --bind-to none --pernode --tag-output bash -lc '
    export OMP_NUM_THREADS=$(nproc)
    python3 -m pytest '"$TEST"' -xvs
  ' > "$LOG" 2>&1 &
  PID=$!
  echo "iteration=$i pid=$PID log=$LOG started=$(date -u -Iseconds)" >> "$CURRENT"

  wait "$PID"
  rc=$?
  echo "=== iteration $i finished rc=$rc at $(date -u -Iseconds) ===" | tee -a "$SUMMARY"
  echo "iteration=$i pid=$PID log=$LOG finished=$(date -u -Iseconds) rc=$rc" >> "$CURRENT"
  if [ "$rc" -ne 0 ]; then
    echo "--- tail of run_${i}.log ---" | tee -a "$SUMMARY"
    tail -40 "$LOG" | tee -a "$SUMMARY"
  fi
done

echo "NO_HANG_AFTER_${END}_ITERATIONS $(date -u -Iseconds)" | tee -a "$SUMMARY"
