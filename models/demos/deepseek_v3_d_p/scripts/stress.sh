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
  # `tee -a "$LOG"` on the exit line, not a bare echo: the status code has to land
  # INSIDE the per-iteration log, because that is the only thing watch.sh reads. A
  # signal kill (SIGBUS/SIGSEGV) prints no pytest summary line at all, so without
  # this the scan can only guess from a log that stopped growing — which is how the
  # 2026-08-12 SIGBUS crashes all displayed as HANG?.
  bash -c "$ENV_VARS pytest -vs \"$PYTEST_TARGET\" |& tee \"$LOG\"; rc=\${PIPESTATUS[0]}; echo \"TEST_DONE_EXIT=\$rc\" | tee -a \"$LOG\""

  # Post-mortem BEFORE the pkill/reset below, while the host state is still the
  # state that failed. This is the only crash forensics available here: dmesg is
  # root-only on these nodes (dmesg_restrict=1), so the driver's own message —
  # e.g. "pin_user_pages_longterm failed: -14" — cannot be captured from a run.
  RC=$(grep -oE 'TEST_DONE_EXIT=[0-9]+' "$LOG" 2>/dev/null | tail -1 | cut -d= -f2)
  if [ -n "$RC" ] && [ "$RC" -ne 0 ]; then
    crash_snapshot "$LOG_DIR" "$i" "$RC" "$LOG"
    echo "### non-zero exit $RC ($(sig_name "$RC")) — snapshot: $(printf '%s/crash_%02d.txt' "$LOG_DIR" "$i")"
  fi

  pkill -9 -f pytest 2>/dev/null || true
  pkill -9 -f test_prefill 2>/dev/null || true
  sleep 2
done

echo ""
echo "############################################################"
printf "###  ALL %d DONE @ %s\n" "$LOOP" "$(date)"
echo "############################################################"
