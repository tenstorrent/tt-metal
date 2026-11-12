#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# h3/power-profile.sh: box power profile of one H3 load on this box (to check synthetic H3's power
# profile against the real H3). Samples with the reliability telemetry logger at about 1 s (chip W from the BMC per-ASIC
# power sensors, per-feed HSC VIN/IOUT/PIN, BMC PSU and power sensors; sideband reads only), runs the load, then writes the profile.
#   power-profile.sh --log <dir> [--idle-s 120] -- <load command...>    any load: synthetic h3.sh, or the real H3 soak
#   power-profile.sh --log <dir> --synthetic [--minutes 20]             h3.sh full mode, then random mode, same window
#   power-profile.sh --log <dir> --report-only [--gens <gens.log>]      profile of telemetry already in <dir>/telemetry
# Output in <dir>: telemetry/ (raw), load.log, profile.csv (1 s: box kW, chip kW, per-tray A and W), profile.json
# (mean, p95 and peak box kW, idle kW, per-tray A, gen seconds, duty cycle), profile.png where matplotlib exists.
# Compare two runs: python3 h3/power_profile.py compare <a/profile.json> <b/profile.json>.
# Idle windows (no load) are sampled before and after the load. No reset and no power or limit change here.
# Exit: the load's exit code; 2 usage.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
LOG="" IDLE=120 SYN=0 MIN=20 ONLY=0 GENS=""
while [ $# -gt 0 ]; do
  case $1 in
    --log) LOG=${2:?}; shift ;;
    --idle-s) IDLE=${2:?}; shift ;;
    --synthetic) SYN=1 ;;
    --minutes) MIN=${2:?}; shift ;;
    --report-only) ONLY=1 ;;
    --gens) GENS=${2:?}; shift ;;
    --) shift; break ;;
    *) echo "power-profile.sh: unknown argument $1" >&2; exit 2 ;;
  esac; shift
done
[ -n "$LOG" ] || { echo "power-profile.sh: --log <dir>" >&2; exit 2; }
[ "$SYN" = 1 ] || [ "$ONLY" = 1 ] || [ $# -gt 0 ] || { echo "power-profile.sh: give --synthetic, --report-only or -- <load command>" >&2; exit 2; }
mkdir -p "$LOG"; LOG=$(cd "$LOG" && pwd)
TEL=$LOG/telemetry
log() { echo "[$(date -u +%FT%TZ)] power-profile: $*" | tee -a "$LOG/profile.log"; }
report() {
  local g=("$@")
  python3 "$HERE/power_profile.py" summarize --tel "$TEL" --out "$LOG" --marks "$LOG/marks.tsv" "${g[@]}" 2>&1 | tee -a "$LOG/profile.log"
}
if [ "$ONLY" = 1 ]; then report ${GENS:+--gens "$GENS"}; exit 0; fi

TEL_SH=$REPO/reliability/telemetry.sh
[ -x "$TEL_SH" ] || { log "no reliability/telemetry.sh in $REPO"; exit 2; }
# 1 s rows; HSC sweeps every 1 s (run.sh's own read path, staggered by tray), BMC sensor get every 5 s.
export TM_ROW_S=${TM_ROW_S:-1} TM_HSC_MS=${TM_HSC_MS:-1000} TM_BMC_S=${TM_BMC_S:-5}
mark() { printf '%s\t%s\n' "$(date +%s)" "$*" >> "$LOG/marks.tsv"; "$TEL_SH" mark "$*" --log "$TEL" > /dev/null 2>&1 || true; }
stop_tel() { "$TEL_SH" stop --log "$TEL" >> "$LOG/profile.log" 2>&1 || true; }
trap stop_tel EXIT
trap 'exit 4' TERM INT HUP
"$TEL_SH" start --log "$TEL" --box "$(hostname -s)" --parent $$ >> "$LOG/profile.log" 2>&1
log "telemetry started in $TEL (row ${TM_ROW_S}s, HSC ${TM_HSC_MS}ms, BMC ${TM_BMC_S}s); idle ${IDLE}s"
mark "idle start"; sleep "$IDLE"; mark "idle end"
rc=0; gl=()
if [ "$SYN" = 1 ]; then
  half=$(( MIN * 30 ))
  for m in full random; do
    d=$LOG/h3-$m; mkdir -p "$d"; tab=(); [ $m = random ] && tab=(--table "$HERE/random-table.tsv")
    mark "load start h3-$m"; log "h3.sh --mode=$m for $(( half / 60 )) min"
    # bounded like reliability.sh's H3 steps (stop time + 900 s, timeout -k 60): h3.sh waits for a stopped pytest
    # without a KILL of its own, so this outer timeout ends a teardown that never finishes
    r=0; timeout -k 60 $(( half + 900 )) "$HERE/h3.sh" --mode=$m "${tab[@]}" --stop-at $(( $(date +%s) + half )) --log "$d" >> "$LOG/load.log" 2>&1 || r=$?
    mark "load end h3-$m rc=$r"; log "h3.sh --mode=$m exit $r"; [ "$r" = 0 ] || rc=$r; gl+=(--gens "$d/gens.log")
  done
else
  mark "load start $*"; log "load: $*"
  "$@" >> "$LOG/load.log" 2>&1 || rc=$?
  mark "load end rc=$rc"; log "load exit $rc"
  [ -n "$GENS" ] && gl+=(--gens "$GENS")
fi
mark "idle start"; sleep "$IDLE"; mark "idle end"
stop_tel; trap - EXIT
report "${gl[@]}"
exit "$rc"
