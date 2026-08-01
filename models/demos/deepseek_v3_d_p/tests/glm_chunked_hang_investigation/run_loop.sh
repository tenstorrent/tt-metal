#!/bin/bash
# Reruns the GLM chunked-prefill test (the one that intermittently hangs/crashes at
# layer 77 in CI, see FINDINGS.md) START..END times. Never kills or times out an
# iteration itself -- a hung iteration just sits in `wait` forever. Run
# poller_loop.sh alongside (see README.md) to detect a stall and triage it without
# disturbing the hung process.
#
# Only resets the device (tt-smi -glx_reset) when it detects a PRIOR iteration left
# the fabric wedged (a crash artifact, not a live hang). For a variant that resets
# before every single iteration (closer to CI's fresh-allocation-per-job behavior),
# use run_loop_reset_each.sh instead.
#
# Usage: ./run_loop.sh [START] [END]        (default 1 15)
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
START="${1:-1}"
END="${2:-15}"

echo $$ > "$LOGDIR/runner.pid"
echo "loop started $(date -u -Iseconds), iterations $START..$END" | tee -a "$SUMMARY"

for i in $(seq "$START" "$END"); do
  LOG="$LOGDIR/run_${i}.log"

  # If the previous iteration crashed (not hung) and left the fabric wedged, reset
  # before starting the next one -- this is cleanup of already-dead state, not an
  # intervention on a live/hung process.
  prev_log="$LOGDIR/run_$((i-1)).log"
  if [ -f "$prev_log" ] && grep -q "Timed out while waiting for active ethernet core" "$prev_log"; then
    echo "prior iteration left device wedged, resetting: tt-smi -glx_reset" | tee -a "$SUMMARY"
    tt-smi -glx_reset > "$LOGDIR/glx_reset_before_${i}.log" 2>&1
  fi

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
