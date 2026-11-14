#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/thermal.sh: lib/thermal.py's read-only sampler through every fleet load chunk
# (fl_c7: the bus-bar stage's mix, heavy and long chunks, quick's one chunk). Sourced by lib/fleet.sh (uses OUT, PY, AS,
# say, gd_py and gd_skip from lib/gddr.sh, mg_iso_path from lib/mce_guard.sh, HL_OUT):
#   th_start <dir>   samples in <dir>/thermal.txt every FLEET_THERM_S s (default 5, never below 1). Not started after a
#                    loss or hang the run cannot name for sure (seen=1, TRY_HUNG, FL_HUNG, FL_SUSPECT; lib/gddr.sh's rule)
#                    and never opens a chip in gd_skip, the MCE guard's iso list or a heavy-log hung_chips.txt
#   th_stop          ends it (the chunk's end, a hang before any capture or hold, the MCE guard's stop)
#   th_chunk <dir>   the chunk's AICLK throttle lines;  th_summary   the run's peaks and hot spots (all chunks)
# FLEET_THERM=0 turns it off. Nothing here changes the RESULT or the exit code.
TH_PID="" TH_DIR=/tmp
th_start() {
  local d=$1 py iv iso=/dev/null why=""
  [ "${FLEET_THERM:-1}" = 0 ] || [ -n "${TRAY_STUB:-}" ] && return 0
  [ "${seen:-0}" = 1 ] && why="a chip was lost in this run"
  [ -n "${TRY_HUNG:-}" ] && why="a try hung in this run"
  [ -n "${FL_HUNG:-}" ] && why="a stage hung in this run"
  [ -n "${FL_SUSPECT[*]+x}" ] && why="a link pair hung in this run"
  [ "${SYS:-/sys}" != /sys ] && [ -z "${TRAY_GDDR_SYS:-}" ] && return 0   # a stand-in sysfs (tests): the real chips are not this run's
  [ -n "$why" ] && { echo "# not sampled: $why (no chip opened)" > "$d/thermal.txt"; return 0; }
  [ -n "${TH_PY:-}" ] || { declare -F gd_py > /dev/null && TH_PY=$(gd_py); }; py=${TH_PY:-}; [ -n "$py" ] || { echo "# not sampled: no python with pyluwen" > "$d/thermal.txt"; return 0; }
  iv=$(awk -v s="${FLEET_THERM_S:-5}" 'BEGIN { if (s !~ /^[0-9]*\.?[0-9]+$/ || s + 0 < 1) s = 1; print s }')
  declare -F mg_iso_path > /dev/null && iso=$(mg_iso_path)
  rm -f "$d/thermal_stop"; : > "$d/thermal.txt"
  local sk=(); declare -F gd_skip > /dev/null && read -ra sk <<< "$(gd_skip)"
  # TH_PARENT: the sampler ends by itself when run.sh is gone (killed: no th_stop; a KILL of sudo leaves its child)
  ${AS[@]+"${AS[@]}"} env TH_PARENT=$$ "$py" "$(dirname "${BASH_SOURCE[0]}")/thermal.py" sample - "$iv" "$d/thermal_stop" \
    "$d/thermal_skip" "$iso" "${HL_OUT:-}" "$d/hang" ${sk[@]+"${sk[@]}"} >> "$d/thermal.txt" 2>/dev/null < /dev/null &
  TH_PID=$! TH_DIR=$d
}
th_stop() {   # the stop file first (the sampler checks it every 0.2 s, also through sudo -u), then TERM, then KILL.
  # th_stop now (a hang or loss, the guard's clock running): the stop file and TERM, no wait (a reaper reaps it)
  [ -n "$TH_PID" ] || return 0
  local i; touch "$TH_DIR/thermal_stop" 2>/dev/null
  if [ "${1:-}" = now ]; then kill "$TH_PID" 2>/dev/null; TH_PID=""; return 0; fi
  for i in $(seq 20); do kill -0 "$TH_PID" 2>/dev/null || break; [ "$i" = 10 ] && kill "$TH_PID" 2>/dev/null; sleep 0.5; done
  kill -9 "$TH_PID" 2>/dev/null; wait "$TH_PID" 2>/dev/null; TH_PID=""
}
th_chunk() { [ -s "$1/thermal.txt" ] && grep -qv '^#' "$1/thermal.txt" && python3 "$(dirname "${BASH_SOURCE[0]}")/thermal.py" chunk "$1/thermal.txt" "${1#"${FL_OUT:-}"/}"; return 0; }
th_summary() {   # every chunk's thermal.txt with a sample; none: the first chunk's reason it was not sampled
  local f=() t n="" d=${FL_OUT:-/tmp}
  for t in "$d"/*/thermal.txt "$d"/*/*/thermal.txt; do
    [ -s "$t" ] || continue; grep -qv '^#' "$t" && f+=("$t") || n=${n:-$(sed -n 's/^# //p;q' "$t")}; done
  if [ ${#f[@]} -eq 0 ]; then [ -n "$n" ] && echo "THERMAL: $n" | tee "$d/thermal_summary.txt"; return 0; fi
  { python3 "$(dirname "${BASH_SOURCE[0]}")/thermal.py" summary "${f[@]}"
    echo "THERMAL RATES: chip temps every ${FLEET_THERM_S:-5} s, BMC every ${BH_POLL_S:-2} s: a burst shorter than that (a 50 ms power step) is not seen"; } | tee "$d/thermal_summary.txt"; return 0
}
