#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/pv_crash.sh (the rule: 'if the box can't handle the load, that's the box's problem'): a host
# crash or reboot under the power virus is a FAIL of that box, named with the stage and the seconds into the leg.
#   pvc_open <stage> <pv dir> <load s>: the leg's open marker (boot id, stage, leg start), fsync'd, at the leg start
#   pvc_stage <stage>: the same marker, its stage moved on (the glx_reset after the load)
#   pvc_close: the leg ended on this boot: the marker goes
#   pvc_check: a marker left open from another boot means the host went down under that leg: one "PV HOST CRASH FAIL"
#     line (stdout and PVC_CRASH), BERT/MCE captured into <pv dir>/crash/ (dmesg -T, journalctl -k -b -1, the ACPI BERT
#     table where readable, ipmitool sel elist where root or sudo -n works), the marker moved there (reported once); rc 1.
#     A marker from this boot (the run was killed, the host stayed up) is set aside with a note, never a crash; rc 0.
#     A crash is called only on a real reboot read on this host: the marker's boot id came from the same boot id file
#     this check reads (boot_src; a marker without one needs a boot id shaped like the kernel's), and this boot began
#     after the leg start (btime in /proc/stat, FLEET_BTIME in tests; a leg start up to 300 s after btime still counts,
#     for a clock that ran ahead before the crash). Otherwise the marker is set aside, never a crash
#     (a tests/ run's fixture boot id 'boot-1' once called a crash on a host that had not rebooted).
#   pvc_verdict: after fl_verdict, a PVC_CRASH makes the RESULT a FAIL (rc 8), never a pass
# The seconds into the leg are from the last line of <pv dir>/stream/samples.txt, which pv.py fsyncs at least every 2 s.
# Run as a script after a reboot (bash lib/pv_crash.sh check) it does the same capture. No device access.
# PVC_MARK (default /var/tmp/tray-test-pv-open.<uid>, kept over a reboot; with FLEET_BOOT_ID set to a file other than the
# kernel's and no PVC_MARK, a file beside the boot id file, so a test never writes or takes the host's marker), FLEET_BOOT_ID (boot id file, tests),
# FLEET_BTIME (this boot's start in epoch s, tests),
# PVC_SUDO (set, even empty: the prefix for the root reads, tests), PVC_DMESG, PVC_JOURNAL, PVC_BERT, PVC_IPMITOOL (tests).
PVC_CRASH=""
PVC_BOOT_SRC=/proc/sys/kernel/random/boot_id
pvc_f() {
  if [ -n "${PVC_MARK:-}" ]; then echo "$PVC_MARK"
  elif [ -n "${FLEET_BOOT_ID:-}" ] && [ "$FLEET_BOOT_ID" != "$PVC_BOOT_SRC" ]; then echo "$(dirname "$FLEET_BOOT_ID")/tray-test-pv-open.$(id -u)"
  else echo "/var/tmp/tray-test-pv-open.$(id -u)"; fi
}
pvc_src() { echo "${FLEET_BOOT_ID:-$PVC_BOOT_SRC}"; }
pvc_boot() { cat "$(pvc_src)" 2> /dev/null; }
pvc_btime() { if [ -n "${FLEET_BTIME:-}" ]; then echo "$FLEET_BTIME"; else awk '/^btime / { print $2; exit }' /proc/stat 2> /dev/null; fi; }
pvc_t() { date -d "@$1" '+%F %T %Z' 2> /dev/null || echo "@$1"; }
pvc_write() {  # <stage> <dir> <load s> <leg start>: write, fsync, rename, fsync the dir
  local f; f=$(pvc_f)
  { echo "boot_id $(pvc_boot)"; echo "boot_src $(pvc_src)"; echo "stage $1"; echo "dir $2"; echo "load_s $3"; echo "start $4"; echo "run ${OUT:-?}"; } > "$f.tmp.$$" 2> /dev/null \
    && sync -- "$f.tmp.$$" 2> /dev/null; mv -f "$f.tmp.$$" "$f" 2> /dev/null && sync -- "$(dirname "$f")" 2> /dev/null
  return 0
}
pvc_get() { sed -n "s/^$1 //p" "${2:-$(pvc_f)}" 2> /dev/null | head -1; }
pvc_open() { pvc_write "$1" "$2" "$3" "$(date +%s)"; }
pvc_stage() {  # also one samples.txt line, so the seconds into the leg count this stage too
  local d st; [ -s "$(pvc_f)" ] || return 0; d=$(pvc_get dir); st=$(pvc_get start)
  pvc_write "$1" "$d" "$(pvc_get load_s)" "$st"
  [ -d "$d/stream" ] && [[ $st =~ ^[0-9]+$ ]] && echo "$(date +%s) $(( $(date +%s) - st )) - - - $1" >> "$d/stream/samples.txt" && sync -- "$d/stream/samples.txt" 2> /dev/null
  return 0
}
pvc_close() { rm -f "$(pvc_f)"; sync -- "$(dirname "$(pvc_f)")" 2> /dev/null; return 0; }
pvc_capture() {  # <dir>: the BERT/MCE evidence of the boot before this one, read-only
  local d=$1 S=() n
  if [ -n "${PVC_SUDO+x}" ]; then read -ra S <<< "$PVC_SUDO"
  elif [ "$(id -u)" != 0 ] && sudo -n true > /dev/null 2>&1; then S=(sudo -n); fi
  mkdir -p "$d"
  # shellcheck disable=SC2086  # the test overrides are commands with args
  timeout -k 5 60 ${PVC_DMESG:-${S[*]} dmesg -T} > "$d/dmesg_T.txt" 2>&1 || echo "(dmesg -T rc $?)" >> "$d/dmesg_T.txt"
  # shellcheck disable=SC2086
  timeout -k 5 120 ${PVC_JOURNAL:-${S[*]} journalctl -k -b -1 --no-pager} > "$d/journal_k_prev_boot.txt" 2>&1 \
    || echo "(journalctl -k -b -1 rc $?)" >> "$d/journal_k_prev_boot.txt"
  n=${PVC_BERT:-/sys/firmware/acpi/tables/BERT}
  if [ -r "$n" ]; then cp "$n" "$d/BERT.bin" 2> /dev/null
  elif [ ${#S[@]} -gt 0 ] && [ -e "$n" ]; then timeout -k 2 10 "${S[@]}" cat "$n" > "$d/BERT.bin" 2> /dev/null || rm -f "$d/BERT.bin"; fi
  [ -s "$d/BERT.bin" ] || { rm -f "$d/BERT.bin"; echo "ACPI BERT table not read ($n: $([ -e "$n" ] && echo "not readable" || echo "absent"))" > "$d/BERT.txt"; }
  if [ "$(id -u)" = 0 ] || [ ${#S[@]} -gt 0 ] || [ -n "${PVC_IPMITOOL:-}" ]; then
    # shellcheck disable=SC2086
    timeout -k 5 120 ${PVC_IPMITOOL:-${S[*]} ipmitool} sel elist > "$d/sel_elist.txt" 2>&1 || echo "(ipmitool sel elist rc $?)" >> "$d/sel_elist.txt"
  else echo "ipmitool sel elist not read: no root and sudo -n does not work" > "$d/sel_elist.txt"; fi
  grep -ahiE 'BERT|mce:|machine check|Hardware Error' "$d/dmesg_T.txt" "$d/journal_k_prev_boot.txt" 2> /dev/null > "$d/bert_mce_lines.txt"
  return 0
}
pvc_aside() {  # <marker> <why>: not a host crash: the marker is set aside with a note, rc 0
  local m; m="$1.killed-$(date +%Y%m%d-%H%M%S)"; mv -f "$1" "$m"
  echo "power virus leg marker set aside ($m: $(pvc_get stage "$m") of run $(pvc_get run "$m") did not end; $2, so not a host crash)"
}
pvc_check() {
  local f b src bt st dir ld last s el how
  f=$(pvc_f); PVC_CRASH=""; [ -s "$f" ] || return 0
  b=$(pvc_get boot_id "$f"); src=$(pvc_get boot_src "$f"); st=$(pvc_get start "$f"); dir=$(pvc_get dir "$f"); ld=$(pvc_get load_s "$f")
  [[ $st =~ ^[0-9]+$ ]] || st=""
  if [ -n "$b" ] && [ "$b" = "$(pvc_boot)" ]; then pvc_aside "$f" "the marker is from this boot, the host stayed up"; return 0; fi
  # a marker written before boot_src: from the host's own boot id file only when its boot id has the kernel's shape
  [ -z "$src" ] && [[ $b =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] && src=$PVC_BOOT_SRC
  if [ "${src:-?}" != "$(pvc_src)" ]; then
    pvc_aside "$f" "its boot id ${b:-?} came from ${src:-no host boot id file}, not from $(pvc_src) that this check reads: no boot-id change read on this host"; return 0; fi
  bt=$(pvc_btime)   # 300 s slack: a crashed boot's clock that ran a little ahead never hides a real crash
  if [ -n "$st" ] && [[ $bt =~ ^[0-9]+$ ]] && [ "$st" -ge $(( bt + 300 )) ]; then
    pvc_aside "$f" "the leg started at $(pvc_t "$st"), after this boot began at $(pvc_t "$bt"): the host stayed up"; return 0; fi
  [ -n "$dir" ] && mkdir -p "$dir" 2> /dev/null || dir="$f.dir-$(date +%Y%m%d-%H%M%S)"   # the pv dir unreadable: evidence beside the marker
  last=$(awk '$1 ~ /^[0-9]+$/ { e = $1 } END { print e }' "$dir/stream/samples.txt" 2> /dev/null)
  if [ -n "$st" ] && [ -n "$last" ] && [ "$last" -ge "$st" ]; then el=$(( last - st )); how="last sample $el s into the leg, at $(pvc_t "$last")"
  else how="no load sample on disk (down before the first sample, $([ -n "$st" ] && echo "leg start $(pvc_t "$st")" || echo "leg start not read"))"; fi
  s="PV HOST CRASH FAIL: the host went down under the $(pvc_get stage "$f") (run $(pvc_get run "$f"), ${ld:-?} s load, leg start $([ -n "$st" ] && pvc_t "$st" || echo "?")): $how; boot ${b:-?} -> $(pvc_boot); BERT/MCE in $dir/crash"
  pvc_capture "$dir/crash"
  [ -s "$dir/crash/bert_mce_lines.txt" ] && s="$s ($(grep -c . "$dir/crash/bert_mce_lines.txt") BERT/MCE line(s))"
  echo "$s" > "$dir/crash/report.txt" 2> /dev/null
  mv -f "$f" "$dir/crash/marker.txt" 2> /dev/null || mv -f "$f" "$f.reported-$(date +%Y%m%d-%H%M%S)"
  PVC_CRASH=$s; echo "$s"; return 1
}
pvc_verdict() {  # after fl_verdict: a host crash under an earlier power virus leg is a FAIL of this box, never a pass
  [ -n "${PVC_CRASH:-}" ] || return 0
  local now=${FL_RESULT:-}; [ -z "$now" ] && [ "${RK_RC:-0}" != 0 ] && now=${RK_WHAT:-}
  FL_RESULT="$PVC_CRASH${now:+; this run: $now}"
  case ${RK_RC:-0} in 0|9) RK_RC=8 ;; esac
  return 0
}
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case ${1:-} in check) pvc_check; exit $? ;; *) echo "usage: $0 check   (after a reboot: report and capture a power virus leg left open)" >&2; exit 2 ;; esac
fi
