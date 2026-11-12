#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/long.sh: run.sh --mode=long (the ask, verbatim: 'add a mode=long that does
# the 3 regular 2 full runs back to back and gives a summary at the end.'). It only chains whole runs: legs 1-3 are plain
# run.sh, legs 4-5 run.sh --mode=full, each re-invoked as its own run (its own gates, health check, C2/C9 glx_reset with
# rescan and recount, recovery, capture and EXIT cleanup) into its own results dir, so plain's and full's stages and step
# lists stay their own (goals 16, 19, 25). It stops early only when a leg could not bring the box healthy (rc 6), the
# host became busy (rc 10) or a signal stopped it; the summary names that leg and why. The EXIT trap prints the summary
# (fleet/long_summary.py): per leg mode, verdict, duration, dir; faults per tray and BDF with k/5; the HSC lines; one
# 'VERDICT:' line for the whole long run; and last one combined 'Repair recommendation:' line. Exit: 6, 10 or 143 as
# above, else 8 when a leg FAILs, 9 when a leg is INCOMPLETE, 0 only when all 5 legs pass.
#   bash lib/long.sh [run.sh options other than --mode]   (run.sh --mode=long execs it from its own dir)
set -euo pipefail
LG_DIR=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)
LG_DRY=0; for a in "$@"; do [ "$a" = --dry-run ] && LG_DRY=1; done
LG_MODES=(quick quick quick full full)
lg_args() { [ "$1" = full ] && echo --mode=full || true; }   # leg mode -> its run.sh option (quick = no --mode)
if [ "$LG_DRY" = 1 ]; then   # each leg's own dry run, one block per leg (tests/test_mode_equal.sh compares them)
  for i in 1 2 3 4 5; do
    m=${LG_MODES[$((i - 1))]}; echo "leg $i/5: run.sh $(lg_args "$m") $*" | tr -s ' '
    # shellcheck disable=SC2046
    bash "$LG_DIR/run.sh" $(lg_args "$m") "$@"
  done
  exit 0
fi
LG_OUT=${TRAY_RUN_OUT:-$PWD/run_$(hostname -s)_long_$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "${LG_OUT:?}"; LG_TSV=$LG_OUT/legs.tsv; : > "$LG_TSV"
LG_PID=""; LG_TP=""; LG_STOP=""
lg_sig() {   # TERM/INT: TERM to our own running leg only (its pid), then the summary names the leg. Always TERM: a leg
  # started with & in this non-interactive bash ignores INT, so Ctrl-C reaches it only this way (run.sh's TERM path
  # reaps its children and reports INCOMPLETE)
  LG_STOP="stopped by SIG$1"; [ -n "$LG_PID" ] && kill -TERM "$LG_PID" 2> /dev/null || true
}
trap 'lg_sig TERM' TERM; trap 'lg_sig INT' INT
lg_end() {
  local rc=$?
  trap - TERM INT
  [ -n "$LG_TP" ] && kill "$LG_TP" 2> /dev/null; [ -n "$LG_TP" ] && wait "$LG_TP" 2> /dev/null
  local src=0
  timeout 60 python3 "$LG_DIR/fleet/long_summary.py" "$LG_TSV" || src=$?
  case $src in 0|8|9) ;; *) src=9
    echo "VERDICT: INCOMPLETE (the long-mode summary failed)"
    echo "Repair recommendation: unknown: the long-mode summary failed; see each leg's own last line in $LG_OUT" ;; esac
  [ "$rc" = 0 ] && rc=$src   # a normal end exits with the long run's VERDICT (8 FAIL, 9 INCOMPLETE); 6, 10, 143 stay
  exit "$rc"
}
trap lg_end EXIT
echo "run.sh --mode=long: legs plain, plain, plain, full, full back to back; results in $LG_OUT"
rc=0
for i in 1 2 3 4 5; do
  m=${LG_MODES[$((i - 1))]}; d=$LG_OUT/leg$i-$m; log=$LG_OUT/leg$i-$m.log; : > "$log"
  echo "=== leg $i/5: run.sh $(lg_args "$m") $* (results $d)" | tr -s ' '
  t0=$(date +%s)
  # shellcheck disable=SC2046
  TRAY_RUN_OUT=$d bash "$LG_DIR/run.sh" $(lg_args "$m") "$@" > "$log" 2>&1 &
  LG_PID=$!
  tail -n +1 --pid="$LG_PID" -f "$log" & LG_TP=$!
  rc=0; wait "$LG_PID" || rc=$?
  while kill -0 "$LG_PID" 2> /dev/null; do rc=0; wait "$LG_PID" || rc=$?; done   # a trap ended the wait early: the leg's own rc
  wait "$LG_TP" 2> /dev/null || true; LG_TP=""; LG_PID=""
  why=""
  case $rc in
    6) why="leg $i could not bring the box healthy (exit 6: chips still missing after its start reset)" ;;
    10) why="leg $i: the host became busy (exit 10)" ;;
  esac
  [ -n "$LG_STOP" ] && why="leg $i $LG_STOP"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$i" "$m" "$rc" "$t0" "$(date +%s)" "$d" "$log" "$why" >> "$LG_TSV"
  [ -n "$why" ] && break
done
[ -n "$LG_STOP" ] && exit 143
case $rc in 6|10) exit "$rc" ;; esac
exit 0
