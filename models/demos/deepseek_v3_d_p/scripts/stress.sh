#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Outer loop: each iteration does `tt-smi -glx_reset` then a foreground pytest run
# (no timeout — stays alive on hang for manual debug). Per-run log: <log_dir>/log_NN.
#
# Usage: stress.sh [log_name] [loop_count]

source "$(dirname "$0")/common.sh" "$@"
set -u

echo "TT_METAL_HOME=$TT_METAL_HOME"
echo "MODEL=$MODEL"
echo "LOG_DIR=$LOG_DIR"
echo "LOOP=$LOOP  INNER_ITERS=$INNER_ITERS"
echo "TARGET=$PYTEST_TARGET"

mkdir -p "$LOG_DIR"

# Preflight: catch a stale/mistyped node id up front instead of after a glx reset and a full weight
# load. Collection imports ttnn and can be slow on a busy box, so it is time-boxed — a timeout is a
# warning, not a failure, and only a definite "does not collect" aborts. PREFLIGHT=0 skips it.
if [ "${PREFLIGHT:-1}" != "0" ]; then
  source "$TT_METAL_HOME/python_env/bin/activate"
  cd "$TT_METAL_HOME"
  echo "preflight: collecting node id (up to ${PREFLIGHT_SECS:-180}s; PREFLIGHT=0 to skip)..."
  PF_OUT=$(timeout "${PREFLIGHT_SECS:-180}" bash -c "$ENV_VARS pytest --collect-only -q \"$PYTEST_TARGET\"" 2>&1)
  PF_RC=$?
  if [ "$PF_RC" -eq 124 ]; then
    echo "preflight: timed out — skipping the check and starting the loop anyway" >&2
  elif [ "$PF_RC" -ne 0 ]; then
    echo "ERROR: node id does not collect — check MODEL / the *_ID overrides:" >&2
    echo "  $PYTEST_TARGET" >&2
    tail -5 <<<"$PF_OUT" >&2
    exit 1
  else
    echo "preflight: OK"
  fi
fi

for i in $(seq 1 "$LOOP"); do
  LOG=$(log_for "$LOG_DIR" "$i")
  echo ""
  echo "############################################################"
  printf "###  Run %02d / %d  (%s inner iter)  @ %s\n" "$i" "$LOOP" "$INNER_ITERS" "$(date)"
  echo "###  log: $LOG"
  echo "############################################################"

  source "$TT_METAL_HOME/python_env/bin/activate"
  tt-smi -glx_reset 2>&1 | tail -3

  cd "$TT_METAL_HOME"
  bash -c "$ENV_VARS pytest -vs \"$PYTEST_TARGET\" |& tee \"$LOG\"; echo TEST_DONE_EXIT=\${PIPESTATUS[0]}"

  pkill -9 -f pytest 2>/dev/null || true
  pkill -9 -f test_prefill 2>/dev/null || true
  sleep 2
done

echo ""
echo "############################################################"
printf "###  ALL %d DONE @ %s\n" "$LOOP" "$(date)"
echo "############################################################"
