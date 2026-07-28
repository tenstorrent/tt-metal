#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Shared config + helpers for the DeepSeek-V3 prefill stress scripts.
# Sourced (not executed) by stress.sh / watch.sh / tail.sh / watch_multiple_dirs.sh.
#
# Common positional args: <log_name> [loop_count]

TT_METAL_HOME="${TT_METAL_HOME:-/data/$USER/tt-metal}"

# Make sure TT_METAL_HOME is on PYTHONPATH (only add it if not already present).
case ":${PYTHONPATH:-}:" in
  *":$TT_METAL_HOME:"*) ;;
  *) export PYTHONPATH="$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

LOG_NAME="${1:-deepseek_v3_d_p_log}"
# Loop count: prefer an explicit LOOP env var (used by watch_multiple_dirs.sh,
# whose positional args are all log names), else the positional [loop_count].
LOOP="${LOOP:-${2:-20}}"

# Per-run logs: one log_NN under here per outer iteration.
LOG_DIR="/data/$USER/$LOG_NAME"

# Test selection — single source of truth.
TEST_FILE="$TT_METAL_HOME/models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py"
KFILTER="test_kimi_prefill_transformer_chunked_no_pcc and k27 and mesh-8x4 and L61 and preload0 and chunks20 and iters20"

# Inner-iteration count, derived from the itersNN token in the filter above.
INNER_ITERS=$(grep -oE 'iters?[0-9]+' <<<"$KFILTER" | grep -oE '[0-9]+' | head -1)

# Model + cache paths handed to pytest.
ENV_VARS='TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/moonshotai/Kimi-K2_7-Code-Cache/Kimi-K2_7-Code-Cache-prefill KIMI_K2_7_HF_MODEL=/mnt/models/moonshotai/Kimi-K2_7-Code-dequantized PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/structured_traces/vllm-kimi-k27-codedebug-56320'

# Seconds without log growth before a still-running iteration is flagged STALE.
STALE_SECS="${STALE_SECS:-240}"

# Path of the Nth outer-iteration log (zero-padded): log_for 3 -> <dir>/log_03
log_for() { printf "%s/log_%02d" "$1" "$2"; }

# Scan one log dir over outer iterations 1..LOOP.
# Sets globals: pass fail hang running pending, and the `details` array.
scan_log_dir() {
  local dir="$1"
  pass=0; fail=0; hang=0; running=0; pending=0
  details=()

  local i f next N iter layer mtime now idle elapsed loading progress
  for i in $(seq 1 "$LOOP"); do
    f=$(log_for "$dir" "$i")
    next=$(log_for "$dir" $((i + 1)))
    N=$(printf "%02d" "$i")
    if [ ! -f "$f" ]; then
      ((pending++))
      continue
    fi
    if grep -qE 'smoke test passed|Chunked prefill no-PCC run done|^=+.*1 passed' "$f" 2>/dev/null; then
      elapsed=$(grep -oE '[0-9]+\.[0-9]+s \([0-9:]+\)' "$f" | tail -1)
      details+=("  $N: PASS  $elapsed")
      ((pass++))
    elif grep -qE '^=+.*(1 failed|1 error)' "$f" 2>/dev/null; then
      details+=("  $N: FAIL")
      ((fail++))
    else
      # Single-shot test logs "Starting iteration:"; the chunked no-PCC test logs
      # "iter N done (C chunks) in ...s" once per completed outer iteration.
      iter=$(grep -cE 'Starting iteration:|iter [0-9]+ done \([0-9]+ chunks\)' "$f" 2>/dev/null)
      layer=$(grep -oE 'forward_layer_[0-9]+_(start|end)' "$f" 2>/dev/null | tail -1)
      mtime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
      now=$(date +%s)
      idle=$((now - mtime))

      # Before the forward loop starts there are no forward_layer markers; show
      # which layer's weights are currently being loaded from cache instead.
      progress="$layer"
      if [ -z "$layer" ]; then
        loading=$(grep 'Loaded cache for' "$f" 2>/dev/null | grep -oE 'layer_[0-9]+' | tail -1)
        [ -n "$loading" ] && progress="loading weights $loading"
      fi

      if [ -f "$next" ]; then
        details+=("  $N: HANG?  iter=$iter/$INNER_ITERS  $progress")
        ((hang++))
      elif [ "$idle" -gt "$STALE_SECS" ]; then
        details+=("  $N: STALE ${idle}s  iter=$iter/$INNER_ITERS  $progress")
        ((running++))
      else
        details+=("  $N: RUN    iter=$iter/$INNER_ITERS  $progress  (idle ${idle}s)")
        ((running++))
      fi
    fi
  done
}
