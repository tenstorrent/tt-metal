#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/psu_fast.sh: per-PSU input power at a fast rate beside the bus-bar stage (and the power
# virus step once it lands), to see whether the plug draw spikes at load onsets and how evenly the PSUs share it. Sourced
# by lib/fleet.sh: functions only. Read-only: the BMC's Power_PSU<n> and Power_Total sensors through `ipmitool -S<sdr cache>
# sensor reading` (Get Sensor Reading; no PMBus to the PSUs: on one site's boxes the BMC refuses master write-read to them; no write,
# no CLEAR_FAULTS). On a Quanta BMC one sweep of the five sensors takes ~0.15 s; the BMC refreshes them every
# ~1-1.5 s in 20 W steps (measured on one box), so reading faster than FLEET_PSU_FAST_S 0.5 adds nothing.
#   pf_prep               early (fl_start, after fl_hsc_latch_start's reads and CLEAR_FAULTS, beside C2): finds the route and an
#                         SDR cache in the background (a cold `sdr dump` takes ~1 min), so the stage's first onset is sampled;
#                         result in $FL_OUT/psu/fast.route. Never a dump inside a load stage: a route not ready by pf_start
#                         (prep still running after PF_ROUTE_WAIT_S, or no prep and no SDR cache) is off for that stage
#   pf_start <dir>        sampler in its own process group -> <dir>/psu_fast.txt: "# names <sensor>...", then one
#                         "<epoch start> <epoch end> <W>..." row per sweep (NA: not read) every FLEET_PSU_FAST_S s (>= 0.2)
#   pf_mark <label>       "# mark <epoch> <label>" in the running sampler's file (a part of the stage starts)
#   pf_stop               ends it; pf_reap (run.sh's EXIT trap) ends the sampler and pf_prep, by process group id
#   pf_report <dir> <what> [<power.txt>]   fleet/psu_fast.py's "PSU FAST" lines: peak and mean per PSU and in total, the
#                         largest single-PSU share, the BMC Power_Total, each load onset against the tray sum/0.94 + 1.5 kW
# No route (no ipmitool, no root or sudo -n for it, no SDR cache, no Power_PSU sensor): one "PSU FAST off: <why>" line.
# Report only: never a FAIL, WARN, stop or cap. FLEET_PSU_FAST=0 turns it off. Each sweep and the route's check read go
# through hsc_i2c where lib/hsc_check.sh is loaded (overlapping in-band IPMI sets the HSCs' CML).
PF_PID=""; PF_PREP_PID=""; PF_FILE=""
PF_NAMES=${PF_NAMES:-Power_PSU0 Power_PSU1 Power_PSU2 Power_PSU3 Power_Total}
pf_on() { [ "${FLEET_PSU_FAST:-1}" != 0 ]; }
pf_interval() { awk -v s="${FLEET_PSU_FAST_S:-0.5}" 'BEGIN { if (s !~ /^[0-9]*\.?[0-9]+$/ || s + 0 < 0.2) s = 0.2; printf "%d", s * 1000000 }'; }
pf_route() {  # "ok <sdr cache> <ipmitool words>" or "off <why>" on stdout; PF_NODUMP=1 (inside a stage): no sdr dump
  local ip=() c b="" n=0 o="" lk=(); declare -F hsc_i2c > /dev/null && lk=(hsc_i2c)
  if [ -n "${TRAY_REPRO_PSU_IPMITOOL:-}" ]; then read -ra ip <<< "$TRAY_REPRO_PSU_IPMITOOL"
    command -v "${ip[0]}" > /dev/null 2>&1 || { echo "off ipmitool not found (${ip[0]})"; return; }
  elif ! command -v ipmitool > /dev/null 2>&1; then echo "off ipmitool not found"; return
  elif [ "$(id -u)" = 0 ]; then ip=(ipmitool)
  else ip=(sudo -n ipmitool); fi
  { read -r b < /proc/sys/kernel/random/boot_id; } 2> /dev/null
  for c in "${PF_SDR:-${b:+/tmp/sdr.cache.$b}}" "$FL_OUT/psu/sdr.cache"; do   # reliability/telemetry.py's per-boot cache when there, else our own dump
    [ -n "$c" ] || continue
    if [ "$c" = "$FL_OUT/psu/sdr.cache" ] && [ "${PF_NODUMP:-0}" = 1 ]; then
      [ -s "$c" ] && [ -r "$c" ] || { echo "off route not ready before the stage"; return; }   # a dump takes ~1 min of in-band IPMI: never in a load stage
    elif [ "$c" = "$FL_OUT/psu/sdr.cache" ]; then mkdir -p "${c%/*}"
      [ "${ip[0]}" = sudo ] && ! timeout --foreground -k 2 5 sudo -n -l ipmitool sdr dump "$c" > /dev/null 2>&1 \
        && { echo "off no root, and sudo -n -l does not allow ipmitool sdr dump"; return; }   # the exact command, never sudo -n true
      rm -f "$c"; timeout --foreground -k 5 "${PF_DUMP_S:-180}" "${ip[@]}" sdr dump "$c" > /dev/null 2>&1
      [ -s "$c" ] || { echo "off ipmitool sdr dump wrote no SDR cache"; return; }
    elif [ ! -s "$c" ] || [ ! -r "$c" ]; then continue; fi
    # shellcheck disable=SC2086  # PF_NAMES: one word per sensor
    [ "${ip[0]}" = sudo ] && ! timeout --foreground -k 2 5 sudo -n -l ipmitool "-S$c" sensor reading $PF_NAMES -c > /dev/null 2>&1 \
      && { echo "off no root, and sudo -n -l does not allow ipmitool sensor reading"; return; }
    # shellcheck disable=SC2086
    o=$("${lk[@]}" timeout --foreground -k 2 10 "${ip[@]}" "-S$c" sensor reading $PF_NAMES -c 2> /dev/null)   # one at a time with the HSC reads
    n=$(grep -cE '^Power_PSU[0-9]+,[0-9]' <<< "$o")
    [ "$n" -gt 0 ] && break   # a stale per-boot cache with no Power_PSU: our own dump next
  done
  [ "$n" -gt 0 ] || { echo "off no Power_PSU sensor reads ($(head -c 120 <<< "$o" | tr '\n' ' '))"; return; }
  echo "ok $c ${ip[*]}"
}
pf_prep() {  # once, early: the route and SDR cache, in the background (its own process group)
  pf_on || return 0; [ -n "$PF_PREP_PID" ] && return 0
  local m=; mkdir -p "$FL_OUT/psu"; rm -f "$FL_OUT/psu/fast.route"
  case $- in *m*) m=1 ;; esac; set -m
  ( trap - USR1 INT TERM EXIT; pf_route > "$FL_OUT/psu/fast.route.tmp"; mv -f "$FL_OUT/psu/fast.route.tmp" "$FL_OUT/psu/fast.route" ) < /dev/null > /dev/null 2>&1 &
  PF_PREP_PID=$!; [ -n "$m" ] || set +m
  return 0
}
pf_loop() {  # <file> <main pid> <prep pid>: the sampler's body (pf_start's subshell)
  local f=$1 main=$2 r="" ip=() c t0 t1 o v l el iv w=$(( ${PF_ROUTE_WAIT_S:-20} * 5 )); iv=$(pf_interval)
  local -A got
  if [ -n "$3" ] || [ -e "$FL_OUT/psu/fast.route" ]; then   # pf_prep's answer: wait for it, bounded
    while [ ! -s "$FL_OUT/psu/fast.route" ] && [ ! -e "$f.stop" ] && [ "$w" -gt 0 ] && { [ -z "$3" ] || kill -0 "$3" 2> /dev/null; }; do sleep 0.2; w=$(( w - 1 )); done
    read -r r < "$FL_OUT/psu/fast.route" 2> /dev/null
    [ -n "$r" ] || { [ -n "$3" ] && pf_kill "$3"; r="off route not ready before the stage"; }   # prep's dump still running: end it, never dump in the stage
  fi
  [ -n "$r" ] || r=$(PF_NODUMP=1 pf_route)
  case $r in "ok "*) ;; *) echo "# off ${r#off }" >> "$f"; return 0 ;; esac
  r=${r#ok }; c=${r%% *}; read -ra ip <<< "${r#* }"
  local lk=(); declare -F hsc_i2c > /dev/null && lk=(hsc_i2c)
  echo "# names $PF_NAMES" >> "$f"
  while kill -0 "$main" 2> /dev/null && [ ! -e "$f.stop" ]; do
    t0=${EPOCHREALTIME/,/.}
    # shellcheck disable=SC2086
    o=$("${lk[@]}" timeout --foreground -k 2 5 "${ip[@]}" "-S$c" sensor reading $PF_NAMES -c 2> /dev/null)
    t1=${EPOCHREALTIME/,/.}; got=()
    while IFS=, read -r l v _; do [[ $v =~ ^[0-9]+(\.[0-9]+)?$ ]] && got[$l]=$v; done <<< "$o"
    l="$t0 $t1"; for v in $PF_NAMES; do l+=" ${got[$v]:-NA}"; done; echo "$l" >> "$f"
    el=$(( ${t1/./} - ${t0/./} )); [ "$el" -lt "$iv" ] && sleep "$(printf '%d.%06d' $(( (iv - el) / 1000000 )) $(( (iv - el) % 1000000 )))"
  done
}
pf_start() {  # <dir>: the sampler, in its own process group (no tty needed)
  pf_on || return 0; pf_stop
  local d=$1 m=; mkdir -p "$d"; PF_FILE=$d/psu_fast.txt; : > "$PF_FILE"; rm -f "$PF_FILE.stop"
  case $- in *m*) m=1 ;; esac; set -m
  ( trap - USR1 INT TERM EXIT; pf_loop "$PF_FILE" "$$" "$PF_PREP_PID" ) < /dev/null > /dev/null 2>&1 &
  PF_PID=$!; [ -n "$m" ] || set +m
  return 0
}
pf_mark() { [ -n "$PF_PID" ] && [ -n "$PF_FILE" ] && echo "# mark ${EPOCHREALTIME/,/.} $*" >> "$PF_FILE"; return 0; }
pf_kill() {  # <pid>...: our own process groups (pf_start, pf_prep): TERM (sudo passes it on), up to 3 s, then KILL
  local p t; for p; do [ -n "$p" ] || continue; kill -TERM -- "-$p" 2> /dev/null || continue
    t=$(( $(date +%s) + 3 )); while kill -0 "$p" 2> /dev/null && [ "$(date +%s)" -lt "$t" ]; do sleep 0.1; done
    kill -KILL -- "-$p" 2> /dev/null; wait "$p" 2> /dev/null; done
  return 0
}
pf_stop() {
  [ -n "$PF_PID" ] || return 0
  : > "$PF_FILE.stop"; local t=$(( $(date +%s) + 6 ))   # one sweep ends on its own (at most its 5 s timeout)
  while kill -0 "$PF_PID" 2> /dev/null && [ "$(date +%s)" -lt "$t" ]; do sleep 0.1; done
  pf_kill "$PF_PID"; rm -f "$PF_FILE.stop"; PF_PID=""; return 0
}
pf_reap() { pf_kill "$PF_PID" "$PF_PREP_PID"; PF_PID=""; PF_PREP_PID=""; return 0; }   # run.sh's EXIT trap
pf_report() {  # <dir> <what> [<power.txt>]: the summary lines on stdout; nothing when the sampler did not run
  pf_on || return 0; [ -e "$1/psu_fast.txt" ] || return 0
  python3 "$FL_DIR/psu_fast.py" "$1/psu_fast.txt" "$2" ${3:+"$3"} 2>&1 | grep -a '^PSU FAST ' || echo "PSU FAST $2: off: the summary did not run"
}
