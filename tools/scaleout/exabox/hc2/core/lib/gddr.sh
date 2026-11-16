#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/gddr.sh: per-chip GDDR telemetry in every run (lib/gddr_telem.py reads it, read-only). Sourced by run.sh
# (uses OUT, PY, AS, say, TRAY_STUB):
#   gd_read start    after the start reset (a reset clears the counters): one "GDDR start <bdf> ..." line per chip
#   gd_read end      before the end reset: the same per chip, then the per-chip rise ("GDDR <bdf> (tray N): corr ...").
#                    No hung or lost chip is opened (an MMIO or ARC read of one can give a fatal MCE that no timeout
#                    stops, lib/mce_guard.sh): the BDFs in MG_HUNG, MG_REMOVED, HL_HUNG and every heavy-log snapshot's
#                    hung_chips.txt are skipped by name; after a loss or hang the run cannot name for sure (seen=1,
#                    TRY_HUNG, a --mode=repro CCL hang, a fleet stage hang FL_HUNG or a C5 link hang FL_SUSPECT) the
#                    whole end read is skipped. The bus-bar stage keeps its hang in FL_HUNG for the verdict even after a
#                    recovery glx_reset; the read goes on only when lib/fleet.sh verified that recovery (FL_BB_HUNG_OK:
#                    glx_reset OK, every chip back, no hang after it), the state the later stages open the whole mesh in
# Both reads leave out the chips whose firmware is not ready (FW_NR, lib/fw_check.sh), as every device open does.
#   gd_summary       in the summary: the GDDR WARN lines (new uncorrected errors, a corrected-error rise) and one
#                    "GDDR: ..." line (chips read, max GDDR temp and where); per-chip lines stay in the log and gddr.txt
# No python with pyluwen (bundle boxes without PyPI), a chip that does not answer, a timeout: 'unavailable' and the run
# goes on; nothing here changes the RESULT or the exit code. --stub (fake devices): not read.
# Env: TRAY_GDDR_PY (the python with pyluwen; default the first of the tt-flash venv, /opt/tenstorrent/venv, $PY),
# TRAY_GDDR_S (the read's timeout, default 60 s; the lines read before it are kept), TRAY_GDDR_CHIP_S (one chip's, 10 s).
gd_py() {
  local p
  if [ -n "${TRAY_GDDR_PY:-}" ]; then echo "$TRAY_GDDR_PY"; return 0; fi
  for p in /opt/tenstorrent/tt-tools/tt-flash/.venv/bin/python /opt/tenstorrent/venv/bin/python "${PY:-python3}"; do
    "$p" -c 'import pyluwen' > /dev/null 2>&1 && { echo "$p"; return 0; }; done
  return 0
}
gd_skip() {  # the BDFs this run saw hung or removed (lib/mce_guard.sh, lib/heavy_logs.sh), one line
  local e r=" ${MG_HUNG:-} ${HL_HUNG:-} "   # HL_HUNG: the last snapshot only, so every snapshot's hung_chips.txt too
  for e in ${MG_REMOVED[@]+"${MG_REMOVED[@]}"}; do r+=" ${e%%|*}"; done
  [ -n "${HL_OUT:-}" ] && r+=" $(cat "$HL_OUT"/*/hung_chips.txt 2>/dev/null | awk '{ print $1 }' | tr '\n' ' ')"
  echo $(echo $r | tr ' ' '\n' | sed 's/^0000://' | sort -u)
}
gd_read() {  # <start|end>
  local f=${OUT:-/tmp}/gddr_$1.txt py rc=0 l skip=() nr=() why=""
  if [ -n "${TRAY_STUB:-}" ]; then : > "$f"; say "GDDR telemetry ($1): not read (--stub, fake devices)"; return 0; fi
  if [ "$1" = end ]; then
    read -ra skip <<< "$(gd_skip)"
    [ "${seen:-0}" = 1 ] && why="a chip was lost in this run"
    [ -n "${TRY_HUNG:-}" ] && why="a try hung in this run"
    [ -n "${ccl:-}" ] && [ -z "${TRY_HUNG:-}" ] && why="a CCL hang in this run"
    [ -n "${FL_HUNG:-}" ] && [ "$FL_HUNG" != "${FL_BB_HUNG_OK:-}" ] && why="a stage hung in this run (${FL_HUNG:0:120})"
    [ -n "${FL_SUSPECT[*]+x}" ] && why="a link pair hung in this run (${FL_SUSPECT[*]})"
    [ -z "$why" ] && [ -n "${FL_HUNG:-}" ] && say "GDDR end: the bus-bar stage hang (${FL_HUNG:0:120}) had a verified recovery (glx_reset OK, every chip back): read"
  fi
  read -ra nr <<< "$(awk '{ sub(/^0000:/, "", $1); print $1 }' <<< "${FW_NR:-}" | paste -sd' ' -)"
  [ -n "${nr[*]}" ] && nr=(--nr "${nr[@]}")
  py=$(gd_py)
  if [ -n "$why" ]; then echo "unavailable $why (not all named): no chip opened before the end reset" > "$f"
    say "GDDR end unavailable: $why (not all named), so no chip is opened before the end reset"
  elif [ -z "$py" ]; then echo "unavailable no python with pyluwen" > "$f"; say "GDDR telemetry ($1): unavailable (no python with pyluwen)"
  else
    timeout "${TRAY_GDDR_S:-60}" ${AS[@]+"${AS[@]}"} "$py" "$(dirname "${BASH_SOURCE[0]}")/gddr_telem.py" read ${skip[@]+"${skip[@]}"} ${nr[@]+"${nr[@]}"} > "$f" 2> "${f%.txt}.err" || rc=$?
    [ "$rc" = 0 ] || echo "unavailable the read ended with rc $rc$([ "$rc" = 124 ] && echo " (timeout ${TRAY_GDDR_S:-60} s)")" >> "$f"
    while IFS= read -r l; do say "GDDR $1 $l"; done < "$f"
    [ -s "$f" ] || say "GDDR $1 unavailable (no output, rc $rc)"
  fi
  if [ "$1" = end ]; then
    python3 "$(dirname "${BASH_SOURCE[0]}")/gddr_telem.py" diff "${OUT:-/tmp}/gddr_start.txt" "$f" > "${OUT:-/tmp}/gddr.txt" 2>&1 || true
    while IFS= read -r l; do say "$l"; done < <(grep -av '^GDDR WARN\|^GDDR: ' "${OUT:-/tmp}/gddr.txt")
  fi
  return 0
}
gd_summary() {  # the WARN lines and the overall line (after gd_read end; nothing before it)
  [ -s "${OUT:-/tmp}/gddr.txt" ] || return 0
  grep -a '^GDDR WARN\|^GDDR: ' "$OUT/gddr.txt"; return 0
}
