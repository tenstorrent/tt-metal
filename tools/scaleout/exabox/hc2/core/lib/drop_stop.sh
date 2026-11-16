#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/drop_stop.sh: a soak or loop driver stops at once on a chip or tray drop (a dead box once looped 4.5 h on 'NOT RUN
# (couldn't start: chips already dropped)'), before any further run or reset, captures the
# failing state, writes one STOP line and exits DS_RC (12). Reads only: no reset, no power action, no CPLD or HSC write.
#   bash lib/drop_stop.sh capture <dir> <why>   the capture below into <dir> (index.txt names each file read or not read)
#   bash lib/drop_stop.sh count                 chips on PCIe (lspci -d 1e52:, one line per chip)
# Sourced: ds_count, ds_trays, ds_drop <expect> <rc> <verdict text>, ds_capture <dir> <why>, ds_stop <file> <run> <what>.
# Capture, each step time-bounded, a step with no route written as 'not read: <why>': lspci -d 1e52: (plain and -vv);
# the kernel log (dmesg, else journalctl -k, as root or where sudo -n -l allows it, else unprivileged dmesg); a tt-smi
# snapshot of the chips left; lib/hsc_dump.sh's per-tray HSC and CPLD dump (hd_dump) and every HSC's latch (hd_latch);
# the BMC SEL (ipmitool sel elist, as root or where sudo -n -l allows it). Tests: TRAY_DROP_LSPCI the lspci command,
# TRAY_DROP_SUDO=0 no sudo route, TRAY_REPRO_IPMITOOL the ipmitool command (also hsc_dump.sh's).
DS_RC=12
DS_LIB=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
ds_lspci() { ${TRAY_DROP_LSPCI:-lspci} -d 1e52: "$@" 2> /dev/null; }
ds_count() { ds_lspci | grep -c . || true; }
ds_trays() {   # per tray 'tray <t> <n>/8' for each tray short of 8 chips on PCIe (buses 0x00-3f tray 1, 40-7f 2, c0-ff 3, 80-bf 4)
  ds_lspci | awk '{ b = tolower($1); if (b ~ /^[0-9a-f]+:[0-9a-f]+:/) sub(/^[0-9a-f]+:/, "", b); h = substr(b, 1, 1)
      t = (h ~ /[0-3]/) ? 1 : (h ~ /[4-7]/) ? 2 : (h ~ /[c-f]/) ? 3 : 4; n[t]++ }
    END { for (t = 1; t <= 4; t++) if (n[t] + 0 < 8) printf "%stray %d %d/8", (s++ ? ", " : ""), t, n[t] + 0 }'; }
ds_drop() {   # <expect> <rc> <verdict text>: 0 with DS_WHAT set on a drop (fewer chips on PCIe, or a run that could not start)
  local n; n=$(ds_count); DS_N=$n; DS_WHAT=""
  if [ "$n" -lt "$1" ]; then DS_WHAT="$n/$1 chips on PCIe ($(ds_trays))"; fi
  if [ "$2" = 6 ] || [[ $3 == *"couldn't start"* ]]; then
    DS_WHAT="${DS_WHAT:+$DS_WHAT; }could not start: chips already dropped (exit $2)"; fi
  [ -n "$DS_WHAT" ]; }
ds_stop() {   # <STOP file> <run> <what>: one line '<UTC> STOP run <run>: <what>', appended and synced
  echo "$(date -u +%FT%TZ) STOP run $2: $3" | tee -a "$1"; sync "$1" 2> /dev/null || true; }
ds_can() { [ "$(id -u)" = 0 ] || { [ "${TRAY_DROP_SUDO:-1}" = 1 ] && timeout 10 sudo -n -l "$@" > /dev/null 2>&1; }; }   # root, or sudo -n allows <cmd...>
ds_as() { if [ "$(id -u)" = 0 ]; then DS_AS=(); else DS_AS=(sudo -n); fi; }   # sets DS_AS, the prefix (timeout runs programs only)
ds_step() {   # <file> <s> <cmd...>: one bounded read into <file>; the index line says read or not read
  local f=$1 s=$2 rc=0; shift 2
  timeout -k 5 "$s" "$@" > "$DS_OUT/$f" 2>&1 < /dev/null || rc=$?
  if [ "$rc" = 0 ]; then echo "$f: read ($*)"; else echo "$f: not read: rc $rc ($*, timeout $s s)"; fi >> "$DS_OUT/index.txt"; }
ds_capture() {   # <dir> <why>
  DS_OUT=$1; mkdir -p "$DS_OUT"
  { echo "drop capture: $2"; echo "start: $(date -u +%FT%TZ) on $(hostname -s); reads only, no reset"; } > "$DS_OUT/index.txt"
  ds_step lspci.txt 60 bash -c "${TRAY_DROP_LSPCI:-lspci} -d 1e52:; echo; ${TRAY_DROP_LSPCI:-lspci} -d 1e52: -vv"
  ds_as
  if ds_can dmesg; then ds_step kernel.txt 60 "${DS_AS[@]}" dmesg -T
  elif ds_can journalctl -k -b --no-pager; then ds_step kernel.txt 60 "${DS_AS[@]}" journalctl -k -b --no-pager
  else ds_step kernel.txt 60 dmesg -T; fi
  if command -v tt-smi > /dev/null; then ds_step tt_smi.txt 180 bash -c 'tt-smi -s --snapshot_no_tty 2>&1 || tt-smi -s'
  else echo "tt_smi.txt: not read: tt-smi not on PATH" >> "$DS_OUT/index.txt"; fi
  if [ -f "$DS_LIB/hsc_dump.sh" ]; then
    ds_step hsc_cpld.txt 120 bash -c '. "$1/hsc_dump.sh"; say() { echo "$*"; }; f=$2
      hd_dump "$f" drop "$(printf "%s\n" 1 2 3 4)"; hd_latch "$f" "drop capture"; cat "$f"' _ "$DS_LIB" "$DS_OUT/.hsc"
  else echo "hsc_cpld.txt: not read: lib/hsc_dump.sh not found" >> "$DS_OUT/index.txt"; fi
  local ip=${TRAY_REPRO_IPMITOOL:-ipmitool}
  if command -v "$ip" > /dev/null && ds_can "$ip" sel elist; then ds_step sel.txt 90 "${DS_AS[@]}" "$ip" sel elist
  else echo "sel.txt: not read: no ipmitool, or no root and sudo -n -l does not allow ipmitool sel elist" >> "$DS_OUT/index.txt"; fi
  echo "end: $(date -u +%FT%TZ)" >> "$DS_OUT/index.txt"; sync 2> /dev/null || true; }
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case ${1:-} in
    capture) ds_capture "${2:?capture needs <dir>}" "${3:-drop}" ;;
    count) ds_count ;;
    *) sed -n '4,5s/^# \{0,1\}//p' "$0" >&2; exit 2 ;;
  esac
fi
