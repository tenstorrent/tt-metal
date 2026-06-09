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
echo "LOG_DIR=$LOG_DIR"
echo "LOOP=$LOOP  INNER_ITERS=$INNER_ITERS"

mkdir -p "$LOG_DIR"

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
  bash -c "$ENV_VARS pytest -vs \"$TEST_FILE\" -k \"$KFILTER\" |& tee \"$LOG\"; echo TEST_DONE_EXIT=\${PIPESTATUS[0]}"

  pkill -9 -f pytest 2>/dev/null || true
  pkill -9 -f test_prefill 2>/dev/null || true
  sleep 2
done

echo ""
echo "############################################################"
printf "###  ALL %d DONE @ %s\n" "$LOOP" "$(date)"
echo "############################################################"
