#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Runs the late-atomic matrix on one chip, safest configs first. Stops at the first hang (exit 3)
# unless RESET_CMD is set, in which case it resets and continues.
#   TT_VISIBLE_DEVICES=0 ./run_matrix.sh                  # default chip 0
#   PAIRS=200000 ONLY="probe_flood ctl_late" ./run_matrix.sh
#   RESET_CMD="env -u TT_VISIBLE_DEVICES tt-smi -glx_reset" ./run_matrix.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
TT="${TT_METAL_HOME:-$(git -C "$HERE" rev-parse --show-toplevel)}"
export TT_METAL_HOME="$TT" PYTHONPATH="$TT"
export TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES:-0}"
# shellcheck disable=SC1091
[ -z "${VIRTUAL_ENV:-}" ] && [ -f "$TT/python_env/bin/activate" ] && source "$TT/python_env/bin/activate"
PAIRS="${PAIRS:-1000000}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ROOT="$TT/generated/late_atomic_bench/matrix_$STAMP"
mkdir -p "$ROOT"

# name | args
CONFIGS=(
  "noflood_probe|--flood-kind none --b-mode probe"
  "probe_flood|--b-mode probe"
  "probe_flood_x16|--b-mode probe --a-incs 16 --flood-kind both"
  "ctl_late|--a-barrier --b-mode late_barrier"
  "ctl_full_x16|--a-barrier --b-mode full_barrier --a-incs 16 --flood-kind both"
  "inc_barrier|--b-mode inc_barrier"
  "incn_barrier|--b-mode incn_barrier --b-incs 4"
  "late_barrier|--b-mode late_barrier"
  "full_barrier_x16|--b-mode full_barrier --a-incs 16 --flood-kind both"
  "late_barrier_noc0|--b-mode late_barrier --noc 0"
)

for entry in "${CONFIGS[@]}"; do
  name="${entry%%|*}"
  args="${entry#*|}"
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]]; then continue; fi
  out="$ROOT/$name"
  echo "=== $name: $args (pairs $PAIRS) -> $out"
  # shellcheck disable=SC2086
  python "$HERE/late_atomic_bench.py" $args --pairs "$PAIRS" --tag "$name" --out-dir "$out" > "$out.log" 2>&1
  rc=$?
  grep -h "SUMMARY\|HANG\|stuck\|late A responses\|no late A" "$out.log" | sed 's/^/    /'
  echo "$name rc=$rc" >> "$ROOT/results.txt"
  if [ $rc -eq 3 ]; then
    if [ -n "${RESET_CMD:-}" ]; then
      echo "    hang: resetting with: $RESET_CMD"
      eval "$RESET_CMD" > "$out.reset.log" 2>&1 || { echo "    reset failed"; exit 3; }
    else
      echo "    hang: chip needs a reset before the next config; stopping (set RESET_CMD to continue)"
      exit 3
    fi
  elif [ $rc -ne 0 ]; then
    echo "    error rc=$rc (see $out.log)"
    exit $rc
  fi
done
echo "results: $ROOT/results.txt"
