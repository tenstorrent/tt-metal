#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

declare -F pg_track > /dev/null || . "$(dirname "${BASH_SOURCE[0]}")/pgroups.sh"   # pg_run (device calls under timeout)
# lib/pinpoint.sh: the numeric pinpoint (the ask: 'is there a way to pinpoint what's causing the bad
# numeric results? chip, dram, noc, etc?'). Sourced by run.sh (uses OUT, FL_OUT, PY, AS, say, hb_start, seen, ccl,
# TRY_HUNG, FL_HUNG). Every run, no flag, but only after a NUMERIC FAIL line in numeric.txt (a --mode=repro try),
# the bus-bar step's numeric.txt or the chunk reports (bus-bar mix, heavy, sustained): a healthy run spends nothing here. Before the end reset, numeric_pinpoint.py opens the mesh
# again (no fabric, no ETH link) and runs on all 32 chips at once: first the failing shape again (repeat: the heavy load
# op and the checked C4 matmul, up to 400 launches or 60 s, until the fault comes back; one fault came back 1 in 194),
# then the splits in passes (pinpoint_verdict.py: dram per bank and lane, NOC0 read vs NOC1 write, matmul in L1 vs DRAM,
# heavy C4n matmul with x, w, y in DRAM plus x-in-L1, read-only and write-only arms, a 60 s burst of an x/w/y-in-DRAM
# C4n-size matmul (rw), per-core FPU and SFPU, capped AICLK once),
# until the failing chips are isolated or the time is spent; then pinpoint_verdict.py prints the lines:
#   NUMERIC-PINPOINT <BDF> tray N component=<...> hits k/N (split ...)    one per finding
#   NUMERIC-PINPOINT VERDICT <BDF> tray N: <component[, odd|even lanes] or 'not isolated (0/N ...)'>    one per failing chip
#   NUMERIC-PINPOINT REPEAT <BDF> tray N: fault back within L launches (S s)[, bank, lanes] | wrong from launch 0 ... | not back in ...
#   NUMERIC-PINPOINT LOAD <BDF> tray N: numeric checks under <op>: launches, box kW, chip peak W and pre-fault C, first bad bank/row/col
# Every check runs under the max-load stage's own load (one chip's bank 3 fault showed only there):
# --load-args is fl_load_args heavy, the one load definition, on all 32 chips; per-launch W/C/kW in $OUT/pinpoint.json.load.jsonl.
# TRAY_PINPOINT_LOAD=0: the checks run solo.
# Never changes the run's result or exit code. Skipped (a VERDICT 'not run' line says why) after a chip loss or a held
# hang: the mesh cannot open before the reset. TRAY_PINPOINT=0 turns it off; TRAY_PINPOINT_S (300) caps its wall time,
# mesh open included. Files: $OUT/pinpoint.json (rewritten after each split), pinpoint.log, pinpoint.txt.
pinpoint_numeric() {
  local c7="" qn="" nf=$OUT/pinpoint-numeric.txt
  local lim=${TRAY_PINPOINT_S:-300} why="" t0 rc=0 e la cap sp=${PINPOINT_SPLITS:-}
  RS_PIN_DONE=1   # run.sh's rs_keep stops keeping its time; its own lim is never cut for the run budget
  [ "${TRAY_PINPOINT:-1}" = 0 ] && return 0
  # the chunk reports: the bus-bar stage's CCL-mix chunks and the sustained part's; the ub0 steps' numeric
  if [ -n "${FL_OUT:-}" ]; then c7=$OUT/pinpoint-chunks.txt; cat "$FL_OUT/busbar/report.txt" "$FL_OUT/busbar/heavy/report.txt" "$FL_OUT/c7/report.txt" > "$c7" 2>/dev/null
    qn=$OUT/pinpoint-steps.txt; cat "$FL_OUT/busbar/step/numeric.txt" > "$qn" 2>/dev/null; fi
  cat "$OUT/numeric.txt" ${qn:+"$qn"} > "$nf" 2>/dev/null   # the repro tries and the bus-bar step: all the local matmul
  grep -aqs '^NUMERIC FAIL' "$nf" ${c7:+"$c7"} || { rm -f "$nf"; return 0; }
  t0=$(date +%s)
  if [ "${seen:-0}" = 1 ]; then why="skipped: a chip was lost, the mesh cannot open before the reset"
  elif [ -n "${ccl:-}${TRY_HUNG:-}${FL_HUNG:-}" ]; then why="skipped: a hang is held until the reset"; fi
  # the chunks' in-run captures (NUMERIC-CAPTURE). Every failing chip captured and none recomputed wrong: no
  # re-trigger (the capture is the attribution); all captured, some reproduce: the splits without the repeat split;
  # the chips not attributed only counter-mapped transients: the dram and rw splits
  cap=$(python3 ./pinpoint_verdict.py capture "$nf" "${c7:-/dev/null}")
  if [ -z "$why" ] && [ "$cap" = skip ] && [ "${TRAY_PINPOINT_CAPTURE_SKIP:-1}" = 1 ]; then
    why="skipped: the in-run capture attributed every failing chip and none recomputed wrong"
  elif [ "$cap" = norepeat ] && [ -z "$sp" ]; then sp=dram,mm,heavy,rw,core,clock
  elif [ "$cap" = transient ] && [ -z "$sp" ]; then sp=dram,rw; fi
  if [ -z "$why" ]; then
    if [ "${TRAY_PINPOINT_LOAD:-1}" = 0 ]; then la=off
    elif declare -F fl_load_args > /dev/null; then la=$(fl_load_args heavy)
    else la=""; fi   # numeric_pinpoint.py's '' is ub0_repro's defaults, the same load
    say "numeric pinpoint: $(python3 ./pinpoint_verdict.py targets "$nf" "${c7:-/dev/null}" | awk '{ printf "%s%s tray %s", (NR > 1 ? ", " : ""), $1, $2 }'); up to $lim s (log $OUT/pinpoint.log)"
    hb_start "numeric pinpoint" "$OUT/pinpoint.log"
    # timeout signals its whole process group (sudo's child too); a split left hung is the end reset's. pg_run: a TERM reaps that group. The run's
    # heavy op goes as a flag: AS (sudo -u) may drop TRAY_HEAVY from the environment; the splits go by env
    pg_run timeout -k 15 "$lim" "${AS[@]}" env PINPOINT_SPLITS="$sp" "$PY" ./numeric_pinpoint.py --numeric "$nf" ${c7:+--c7 "$c7"} \
      --result "$OUT/pinpoint.json" --budget-s $(( lim > 60 ? lim - 30 : lim )) --heavy-op "${TRAY_HEAVY:-}" \
      --load-args "$la" \
      > "$OUT/pinpoint.log" 2>&1 < /dev/null || rc=$?
    hb_stop
    case $rc in
      0) ;;
      124|137) why="timed out after $lim s, the splits that ended are kept" ;;
      *) e=$(grep -a -m1 -E 'PINPOINT (error|split .* error)|Error|error:' "$OUT/pinpoint.log" | cut -c1-200)
         why="pinpoint rc $rc${e:+: $e}" ;;
    esac
  fi
  python3 ./pinpoint_verdict.py capture "$nf" "${c7:-/dev/null}" "$OUT/pinpoint.json" > /dev/null
  python3 ./pinpoint_verdict.py report "$nf" "${c7:-/dev/null}" "$OUT/pinpoint.json" "$why" > "$OUT/pinpoint.txt"
  while IFS= read -r e; do say "$e"; done < "$OUT/pinpoint.txt"
  say "numeric pinpoint: $(( $(date +%s) - t0 )) s added to this failing run; $OUT/pinpoint.txt"
  return 0
}
pinpoint_summary() {  # the verdict lines again with the run's last lines (before fl_report or the RESULT block)
  [ -s "$OUT/pinpoint.txt" ] && grep -a '^NUMERIC-PINPOINT VERDICT' "$OUT/pinpoint.txt"
  return 0
}
