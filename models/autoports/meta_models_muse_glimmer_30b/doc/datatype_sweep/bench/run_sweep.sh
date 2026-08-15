#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# One 52-layer build per candidate, one process per candidate.
#
# A process per candidate rather than a loop inside one: each build holds
# ~7.2 GB/device of long-lived DRAM, and a fresh process is the only way to be
# sure a candidate's numbers are not a previous candidate's allocator state.
# The device is opened and closed inside sweep.py, so the commands are
# serialised here rather than run in parallel ($tt-device-usage).
#
# ROUNDS=<n> sets the teacher-forcing repeat count (sweep.py's --rounds).  The
# first pass uses the default 5 to rank and gate; a second pass re-runs the
# candidates that passed at a higher count, because the ranking metric's
# round-to-round spread (~0.2 %) is the same order as the difference between the
# closest passing candidates.  Both passes must use the same count within a
# comparison, so the second pass re-runs every passing candidate, not just the
# contenders.
#
# Usage:  doc/datatype_sweep/bench/run_sweep.sh [config-id ...]
#         ROUNDS=11 doc/datatype_sweep/bench/run_sweep.sh c00-... c01-...
# Watch:  tail -f models/autoports/meta_models_muse_glimmer_30b/doc/datatype_sweep/logs/sweep.log

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # models/autoports/<model>
REPO="$(cd "$ROOT/../../.." && pwd)"
LOGS="$ROOT/doc/datatype_sweep/logs"
mkdir -p "$LOGS"

cd "$REPO" || exit 1

if [ "$#" -gt 0 ]; then
  CONFIGS=("$@")
else
  mapfile -t CONFIGS < <(cd "$ROOT/doc/datatype_sweep/configs" && ls *.json | sed 's/\.json$//' | sort)
fi

ROUNDS=${ROUNDS:-5}
echo "SWEEP_DRIVER candidates: ${CONFIGS[*]} (rounds=$ROUNDS)"
FAILED=()
for config in "${CONFIGS[@]}"; do
  echo "SWEEP_DRIVER ===== $config started $(date -Is) ====="
  if timeout 3600 python "$ROOT/doc/datatype_sweep/bench/sweep.py" --config "$config" --rounds "$ROUNDS" \
      > "$LOGS/sweep_$config.log" 2>&1; then
    grep -E "^SWEEP " "$LOGS/sweep_$config.log" | tail -12
    echo "SWEEP_DRIVER ===== $config ok $(date -Is) ====="
  else
    status=$?
    FAILED+=("$config")
    echo "SWEEP_DRIVER ===== $config FAILED (exit $status) $(date -Is) ====="
    grep -E "^SWEEP |TT_THROW|Error|error:" "$LOGS/sweep_$config.log" | tail -20
  fi
done

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "SWEEP_DRIVER failed candidates: ${FAILED[*]}"
else
  echo "SWEEP_DRIVER all candidates completed"
fi
echo "SWEEP_DRIVER_DONE"
