#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/hsc_check.sh: each tray's two 12 V hot-swap controllers (MP5990, 8-bit
# addresses 0x40 and 0x42), read in-band before and after each load leg (fl_loads: the bus-bar stage and C5, the same
# legs in plain, full and smoke). Sourced by lib/fleet.sh: functions only. Why: one box's tray 4 lost its 12 V
# under load; a dip shorter than any poll still latches the HSC's VIN undervoltage bits, and those latches are what the
# per-second 12 V log (fl_12v) cannot see.
#   Reads: `ipmitool raw 0x06 0x52 <bus> <addr> <n> <code>` (master write-read) with code STATUS_WORD 0x79 (2 B),
#   STATUS_INPUT 0x7C (1 B), READ_VIN 0x88 (2 B, 31.25 mV/LSB), READ_IOUT 0x8C (2 B, 62.5 mA/LSB: the MP5990's tray-input
#   current; READ_IIN reads 0 there, notes/tray-12v-margin-A.md) or VIN_UV_WARN_LIMIT 0x58 (2 B, bits 6:0 x 125 mV), plus
#   HSC_VIN_MIN_REG / HSC_VIN_PEAK_REG when set, and STATUS_CML 0x7E (1 B) only where STATUS_WORD shows CML (bit 1).
#   Fault-record reads (codes from the vendor): FAULT_RECORD 0xFA and FAULT_RECORD_LAST 0xFB (2 B
#   each), STATUS_IOUT 0x7B (1 B) and READ_VOUT 0x8B (2 B, 31.25 mV/LSB) in every leg and run snapshot; IOUT_OC_FAULT_LIMIT
#   0x46 and IOUT_OC_WARN_LIMIT 0x4A (2 B, 62.5 mA/LSB as READ_IOUT, the Linux mp5990 driver's current format) once per
#   run; READ_IOUT_PEAK 0xA6 (2 B, 62.5 mA/LSB) and READ_TEMP_PEAK 0xAF (2 B, deg C), which clear when read, only through
#   hsc_peaks: once before the CLEAR_FAULTS (the peak since the last read), once just before each load leg (this zeroes
#   them) and once right after it (that leg's peak). Peak WARN only, never FAIL: a feed's leg peak above
#   its own IOUT_OC_WARN_LIMIT, or more than HSC_PEAK_PCT (20) % above the median of the other feeds' leg peaks.
#   The run-start peaks (from before the run) join the one 'HSC WARN host: ...; latched before the run: pre-existing' line
#   with the pre-existing latch bits, never one line per feed; a leg's own peak keeps its per-feed line.
#   hsc_rd refuses any other code: of the clear, read-clear and write ranges it allows exactly 0xFA, 0xFB, 0xA6 and 0xAF.
#   CML: STATUS_WORD bit 1 is the PMBus communication/logic fault, not power. On one box
#   (11-02) STATUS_CML read 80h (invalid or unsupported command) on the 5 HSCs whose CML set during a smoke run; none of
#   our codes is unsupported (240 sequential 79h/7Ch/88h reads and 8 58h reads set none), while 40 s of overlapping in-band
#   I2C transactions (two 88h samplers, sdr) set it again: overlapping master write-reads garble the command byte. So
#   every in-band raw 0x06 0x52 of ours (hsc_rd, hsc_clear_faults, fl_power's CPLD read) and fl_power's sensor reads go
#   one at a time through hsc_i2c (flock), and CML is graded on its own: WARN at most, never FAIL, never "none", and never
#   read into VIN_UV_WARN, VIN_UV_FAULT, OFF or IOUT_OC (each graded from its own bit).
#   The one write (on in every mode, no switch): hsc_clear_faults, PMBus CLEAR_FAULTS
#   (03h, a send-byte with no data) once per tray HSC per run, before C2, through the same HSC_IP route as the reads, only
#   after every HSC's STATUS_WORD, STATUS_INPUT, FAULT_RECORD, STATUS_IOUT and peaks are read and logged and only when no HSC reads OFF (one OFF: no clear to any
#   HSC this run, and the output says so; an HSC whose STATUS_WORD did not read gets none, named; a failed clear is named,
#   log only). No other HSC, BMC or CPLD write anywhere: no CLEAR_PEAK, no limit.
#   Min and peak VIN: the MP5990's PMBus map has no READ_VIN_MIN / READ_VIN_PEAK register (the Linux mp5990 driver
#   exposes none; the MPS register export lists none; 0xD1/0xD2 are other parts' registers). So the leg's min and peak
#   come from a READ_VIN + READ_IOUT sampler (hsc_poll_start, every HSC_VIN_POLL_S, 2 s) plus the before and after reads.
#   HSC_VIN_MIN_REG / HSC_VIN_PEAK_REG (e.g. 0xD1): a 2-byte register in the same 31.25 mV/LSB format, read before and
#   after and folded into the leg's min and peak (for a part that has one).
# Grading: FAIL only on VIN_UV_WARN or VIN_UV_FAULT (STATUS_INPUT bits 5, 4) newly set during the run,
#   STATUS_WORD OFF (bit 6) or IOUT_OC (bit 4), or a chip or tray drop (fleet.sh's own lines). A latch set before the run
#   is a pre-existing WARN. The 12 V sag rule and the absolute-VIN floor print HSC WARN only, never FAIL:
#   sag: per HSC and leg, idle VIN = median VIN of the leg's samples at 2 A < IOUT <= 30 A (3 or more), else the read just
#   before the leg (at quiet); for each sample at IOUT >= HSC_SAG_MIN_A (40 A, a re-checked floor),
#   R = (idle VIN - VIN) / IOUT; R max over HSC_SAG_MOHM (7.5 mOhm) is an HSC WARN naming the tray, BDFs and HSC.
#   Floor: min VIN below HSC_VIN_MIN_V (11.0 V) during the leg is an HSC WARN. Min/peak VIN: evidence lines only.
#   Per feed, every leg in every mode: "HSC feed tray N (...) HSC 0x4X during <stage>:"
#   VIN p10 (nearest rank) and median and IOUT median of the samples at IOUT >= HSC_SAG_MIN_A, R = (idle VIN - VIN median) /
#   IOUT median in mOhm, and its gap to the median R of the other feeds with a number. Evidence only, never a WARN or FAIL.
#   No number for an HSC that read OFF before or after the leg; a sample reading 0xFFFF, VIN outside 6-15 V or IOUT over
#   250 A feeds neither this line nor the sag rule.
# Run start (fl_hsc_latch_start, before C2): STATUS_WORD and STATUS_INPUT of every HSC -> fleet/hsc_latch_before.tsv and
#   the run output; each bit already set: an HSC WARN, pre-existing. Then the clear and a re-read ->
#   fleet/hsc_latch_cleared.tsv. Run end (fl_hsc_latch_end, after C9): read again -> fleet/hsc_latch_after.tsv. Graded per
#   bit (VIN_UV_WARN, VIN_UV_FAULT, OFF, IOUT_OC): set at the end and clear before the run, or cleared by
#   CLEAR_FAULTS and set again: an HSC FAIL; set before the run and still set right after the clear (or no clear): the
#   pre-existing WARN only; set at the end with its own before-register unread and no clean read after CLEAR_FAULTS: not
#   graded, with the reason, never a FAIL. fl_hsc_latch_table: the before/after table in fl_report.
# Per HSC after each leg (fl_hsc_after):
#   HSC FAIL tray N (<BDF range>) HSC 0x4X: VIN_UV_WARN|VIN_UV_FAULT latched during <stage> (clear before), or OFF or
#     IOUT_OC set during it. A finding, never a stop: the load goes on. The end read then names that HSC once only.
#   HSC WARN tray N (...) HSC 0x4X: 12 V sag R max over 7.5 mOhm, or min VIN below 11.0 V, during the leg: a warn (the
#     tray line reads PASS tray N (k warn)), never a FAIL and never the run verdict.
#   A UV latch or a low VIN already there before the leg: printed once per tray and HSC per run, "HSC tray N (...) HSC
#     0x4X latched before load: a new dip on this tray cannot be seen until a power cycle clears it", never a finding.
#   No read route (no ipmitool, no root and no sudo -n) or no HSC answering: "HSC check not run: ..." once per run and
#     "HSC latch: not read (<why>); no CLEAR_FAULTS sent", plainly, never a FAIL. An HSC that does not answer one read is "not read".
#     The end summary (fl_report, every mode) then prints "WARN HSC check NOT RUN: <reason>" (no root (sudo -n -l refuses ipmitool raw 0x06 0x52),
#     no ipmitool, no ipmi device (...), no HSC answered, or not reached), never a finding.
# TRAY_REPRO_HSC_IPMITOOL: the ipmitool command to use as is (tests: a fake); unset: ipmitool as root, else sudo -n ipmitool.
HSC_IP=(); HSC_WHY=""; HSC_NR=""; HSC_POLL_PID=""; HSC_LEG=""; HSC_CLEAR_OK=0
HSC_TRAYS="1 2 3 4"; HSC_ADDRS="0x40 0x42"
hsc_bus() { case $1 in 1) echo 0x11 ;; 2) echo 0x13 ;; 3) echo 0x0d ;; 4) echo 0x0f ;; esac; }
hsc_bdfs() { case $1 in 1) echo 01:00.0-08:00.0 ;; 2) echo 41:00.0-48:00.0 ;; 3) echo c1:00.0-c8:00.0 ;; 4) echo 81:00.0-88:00.0 ;; esac; }
hsc_route() {  # sets HSC_IP; rc 2 with the why in HSC_WHY when no read route (no subshell: HSC_IP must stay set)
  HSC_IP=(); HSC_WHY=""; HSC_NR=""   # HSC_NR: the short reason for the end summary's 'WARN HSC check NOT RUN: <reason>'
  if [ -n "${TRAY_REPRO_HSC_IPMITOOL:-}" ]; then read -ra HSC_IP <<< "$TRAY_REPRO_HSC_IPMITOOL"
    command -v "${HSC_IP[0]}" > /dev/null 2>&1 || { HSC_IP=(); HSC_WHY="ipmitool not found (${TRAY_REPRO_HSC_IPMITOOL%% *})"; HSC_NR="no ipmitool"; return 2; }
  elif ! command -v ipmitool > /dev/null 2>&1; then HSC_WHY="ipmitool not found"; HSC_NR="no ipmitool"; return 2
  elif [ "$(id -u)" = 0 ]; then HSC_IP=(ipmitool)
  elif timeout -k 2 5 sudo -n -l ipmitool raw 0x06 0x52 0x11 0x40 0x02 0x79 > /dev/null 2>&1; then HSC_IP=(sudo -n ipmitool)   # the exact read, never sudo -n true (some sites' sudoers allow only ipmitool and tt-smi)
  else HSC_WHY="no root, and sudo -n -l does not allow ipmitool raw 0x06 0x52"; HSC_NR="no root (sudo -n -l refuses ipmitool raw 0x06 0x52)"; return 2; fi
}
# hsc_clear_faults <tray> <addr>: the one PMBus write in tray-test: CLEAR_FAULTS (03h), which resets the
# HSC's status latches and nothing else. Guarded: only while fl_hsc_latch_start has seen no HSC OFF (HSC_CLEAR_OK=1), only
# to HSC 0x40 or 0x42 on a tray 1-4 bus, once per HSC per run (a marker under $FL_OUT/hsc), command byte fixed here.
hsc_clear_faults() {
  local m
  [ "${HSC_CLEAR_OK:-0}" = 1 ] || { echo "hsc_clear_faults: refused (no all-HSCs-not-OFF read before it)" >&2; return 2; }
  case $2 in 0x40|0x42) ;; *) echo "hsc_clear_faults: refused address $2" >&2; return 2 ;; esac
  [ -n "$(hsc_bus "$1")" ] || { echo "hsc_clear_faults: refused tray $1" >&2; return 2; }
  m=$(fl_hsc_f "cleared_${1}_${2}"); [ -e "$m" ] && { echo "hsc_clear_faults: refused (tray $1 HSC $2 already cleared this run)" >&2; return 2; }
  : > "$m"
  hsc_i2c timeout "${HSC_READ_S:-5}" "${HSC_IP[@]}" raw 0x06 0x52 "$(hsc_bus "$1")" "$2" 0x00 0x03 > /dev/null 2>&1
}
hsc_i2c() {  # <cmd...>: run one in-band IPMI command with no other of ours in flight (overlap sets CML)
  local lk=${HSC_I2C_LOCK:-${FL_OUT:-${TMPDIR:-/tmp}}/hsc_i2c.lock}
  if command -v flock > /dev/null 2>&1 && { : >> "$lk"; } 2> /dev/null; then flock -w "${HSC_I2C_WAIT_S:-10}" "$lk" "$@"; else "$@"; fi
}
hsc_rd() {  # <tray> <addr> <code>: the register's value in decimal on stdout, nothing when it does not answer
  local n o b ok=""
  case $3 in 0x79|0x58) n=0x02 ;; 0x7C|0x7c|0x7E|0x7B|0x7b) n=0x01 ;; 0x88|0x8C|0x8c|0x8B|0x8b|0x46|0x4A|0x4a) n=0x02 ;;
    0xFA|0xfa|0xFB|0xfb|0xA6|0xa6|0xAF|0xaf) n=0x02; ok=1 ;;   # fault record and peaks: the only four allowed in the ranges below
    *) if [ -n "$3" ] && { [ "$3" = "${HSC_VIN_MIN_REG:-}" ] || [ "$3" = "${HSC_VIN_PEAK_REG:-}" ]; }; then n=0x02
       else echo "hsc_rd: refused code $3 (not a whitelisted read)" >&2; return 2; fi ;; esac
  case $3 in 0x0[0-3]|0x1[0-9a-fA-F]|0x[eE]*) echo "hsc_rd: refused code $3 (not a read register)" >&2; return 2 ;; esac   # CLEAR_FAULTS 0x03, CLEAR_PEAK etc.
  # HSC_VIN_MIN_REG / HSC_VIN_PEAK_REG: two hex digits only (ipmitool also takes decimal and octal), and never 0xA0-0xAF:
  # the MP5990's read-clear peak registers, where a read would be a hidden second clear (see bin/tray-pmbus-read)
  [[ $3 =~ ^0x[0-9a-fA-F]{2}$ ]] || { echo "hsc_rd: refused code $3 (not 0x plus two hex digits)" >&2; return 2; }
  # the exception holds only for the fixed list above, never for a code that also came in as HSC_VIN_MIN_REG/HSC_VIN_PEAK_REG
  local mr=${HSC_VIN_MIN_REG:-} pr=${HSC_VIN_PEAK_REG:-}; case ${3,,} in "${mr,,}"|"${pr,,}") ok="" ;; esac
  if [ -z "$ok" ] && (( 16#${3#0x} < 0x20 || (16#${3#0x} >= 0xA0 && 16#${3#0x} <= 0xAF) || 16#${3#0x} >= 0xE0 )); then
    echo "hsc_rd: refused code $3 (a clear, read-clear or write register)" >&2; return 2; fi
  o=$(hsc_i2c timeout "${HSC_READ_S:-5}" "${HSC_IP[@]}" raw 0x06 0x52 "$(hsc_bus "$1")" "$2" "$n" "$3" 2> /dev/null) || return 0
  read -ra b <<< "$o"
  [ "${#b[@]}" = $((n)) ] || return 0
  [[ "${b[0]}${b[1]:-}" =~ ^[0-9a-fA-F]+$ ]] || return 0
  if [ $((n)) = 2 ]; then echo $(( 16#${b[0]} | (16#${b[1]} << 8) )); else echo $(( 16#${b[0]} )); fi
}
hsc_snap() {  # <file>: one row per tray and HSC: tray addr STATUS_WORD STATUS_INPUT READ_VIN MIN PEAK FAULT_RECORD
  local t a v sw si vin mn pk fr so vo   # STATUS_IOUT READ_VOUT ("-" = not read); FAULT_RECORD_LAST (last power-off) is read
  # once per run in the latch table, not per leg (KCS costs 110-375 ms a read)
  : > "$1"
  for t in $HSC_TRAYS; do for a in $HSC_ADDRS; do
    sw=$(hsc_rd "$t" "$a" 0x79); si=$(hsc_rd "$t" "$a" 0x7C); vin=$(hsc_rd "$t" "$a" 0x88); mn=""; pk=""
    [ -n "${HSC_VIN_MIN_REG:-}" ] && mn=$(hsc_rd "$t" "$a" "$HSC_VIN_MIN_REG")
    [ -n "${HSC_VIN_PEAK_REG:-}" ] && pk=$(hsc_rd "$t" "$a" "$HSC_VIN_PEAK_REG")
    fr=$(hsc_rd "$t" "$a" 0xFA); so=$(hsc_rd "$t" "$a" 0x7B); vo=$(hsc_rd "$t" "$a" 0x8B)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$t" "$a" "${sw:--}" "${si:--}" "${vin:--}" "${mn:--}" "${pk:--}" \
      "${fr:--}" "${so:--}" "${vo:--}" >> "$1"
  done; done
}
hsc_frdec() {  # <FAULT_RECORD or FAULT_RECORD_LAST, decimal or 0x>: "0x0022 (IOUT_OC_FAULT+bit1)" (from the vendor:
  # bit 4 VIN_UVLO = VIN under 8 V, 0x36 reads 0x40 = 8.0 V; bit 5 IOUT_OC_FAULT = 228/232 A for 200 us); a decode, never a finding
  local v=${1:-} n b d=""
  case $v in ""|-) echo "not read"; return 0 ;; *[!0-9xXa-fA-F]*|0[xX]) echo "$v"; return 0 ;; esac
  n=$((v)); for ((b = 15; b >= 0; b--)); do (( n >> b & 1 )) || continue
    case $b in 4) d+="+VIN_UVLO" ;; 5) d+="+IOUT_OC_FAULT" ;; *) d+="+bit$b" ;; esac; done
  printf '0x%04x' "$n"; [ -z "$d" ] || printf ' (%s)' "${d#+}"; echo
}
hsc_step_line() {  # <before|after> <stage name> <snap> <peaks>: one line of the fault-record reads per feed with the step and time
  local t a sw si vin mn pk fr so vo ip tp s="" P
  declare -A P; while IFS=$'\t' read -r t a ip tp; do P[$t/$a]="$ip $tp"; done < "$4"
  while IFS=$'\t' read -r t a sw si vin mn pk fr so vo; do
    read -r ip tp <<< "${P[$t/$a]:-- -}"
    s+=", t$t/$a $(hsc_frdec "$fr") iout $([ "${so:--}" = - ] && echo - || printf '0x%02x' "$so") peak $([ "$ip" = - ] && echo - || awk -v r="$ip" 'BEGIN { printf "%.1f A", r / 16 }') $([ "$tp" = - ] && echo - || echo "$tp C")"
  done < "$3"
  fl_hsc_say "HSC reads $1 $2 at $(date '+%H:%M:%S %Z') (FAULT_RECORD, STATUS_IOUT, IOUT peak (A6h, 1/16 A), TEMP peak (AFh)): ${s#, }"
}
hsc_peaks() {  # <file>: READ_IOUT_PEAK 0xA6 and READ_TEMP_PEAK 0xAF of every HSC, raw ("-" = not read). Read-clear: this read
  local t a ip tp   # zeroes both, so only fl_hsc_latch_start, fl_hsc_before and fl_hsc_after call it
  : > "$1"
  for t in $HSC_TRAYS; do for a in $HSC_ADDRS; do
    ip=$(hsc_rd "$t" "$a" 0xA6); tp=$(hsc_rd "$t" "$a" 0xAF)
    printf '%s\t%s\t%s\t%s\n' "$t" "$a" "${ip:--}" "${tp:--}" >> "$1"
  done; done
}
hsc_oclim() {  # once per run: IOUT_OC_FAULT_LIMIT 0x46 and IOUT_OC_WARN_LIMIT 0x4A of every HSC -> fleet/hsc/oc_limits (raw)
  local f t a; f=$(fl_hsc_f oc_limits); [ -e "$f" ] && return 0
  for t in $HSC_TRAYS; do for a in $HSC_ADDRS; do
    printf '%s\t%s\t%s\t%s\n' "$t" "$a" "$(hsc_rd "$t" "$a" 0x46)" "$(hsc_rd "$t" "$a" 0x4A)"; done; done | sed 's/\t\t/\t-\t/; s/\t$/\t-/' > "$f.tmp"
  mv "$f.tmp" "$f"
  fl_hsc_say "HSC IOUT OC fault/warn limit A (46h/4Ah, once per run):$(awk -F'\t' '{ printf " t%s/%s %s/%s", $1, $2, ($3 == "-" ? "-" : sprintf("%.1f", $3 * 0.0625)), ($4 == "-" ? "-" : sprintf("%.1f", $4 * 0.0625)) }' "$f")"
}
hsc_peak_judge() {  # <stage name> <leg peaks> [<oc limits>] [pre]: "HSC WARN ..." per feed over its own warn limit or HSC_PEAK_PCT % over
  # the median of the other feeds' leg peaks (WARN only, never FAIL), then one "HSC peak after" line.
  # pre (peaks read at run start, from before the run): one "HSC PRE <item>" line per feed instead, for the caller's one host WARN line
  awk -F'\t' -v st="$1" -v pre="${4:-}" -v pct="${HSC_PEAK_PCT:-20}" -v lmin="${HSC_OC_LIM_MIN_A:-50}" -v fp="$2" -v r1="01:00.0-08:00.0" -v r2="41:00.0-48:00.0" \
      -v r3="c1:00.0-c8:00.0" -v r4="81:00.0-88:00.0" '
    function med(A, n,  i, j, x) { for (i = 2; i <= n; i++) { x = A[i]; for (j = i - 1; j >= 1 && A[j] > x; j--) A[j + 1] = A[j]; A[j + 1] = x }
      return n % 2 ? A[(n + 1) / 2] : (A[n / 2] + A[n / 2 + 1]) / 2 }
    BEGIN { R[1] = r1; R[2] = r2; R[3] = r3; R[4] = r4 }
    FILENAME == fp { k = $1 "\t" $2; K[++n] = k; if ($3 != "-" && $3 != "") P[k] = $3 * 0.0625; TP[k] = $4; next }
    { if ($4 != "-" && $4 != "" && $4 * 0.0625 >= lmin) WL[$1 "\t" $2] = $4 * 0.0625 }
    END {
      for (i = 1; i <= n; i++) { k = K[i]; split(k, p, "\t"); t = p[1]; a = p[2]
        s = s sprintf(" t%d/%s %s", t, a, (k in P) ? sprintf("%.1f", P[k]) : "not read"); c = c sprintf(" t%d/%s %s", t, a, TP[k] == "-" ? "not read" : TP[k] + 0)
        if (!(k in P)) continue
        h = sprintf("tray %d (%s) HSC %s: IOUT peak %.1f A during %s", t, R[t], a, P[k], st); w = ""
        if ((k in WL) && P[k] >= WL[k]) { x = sprintf("at or above its own IOUT_OC_WARN_LIMIT %.1f A (4Ah)", WL[k]); w = w ", " x
          if (!pre) printf "HSC WARN %s, %s; WARN only, not a finding\n", h, x }
        m = 0; delete O; for (j = 1; j <= n; j++) if (j != i && (K[j] in P)) O[++m] = P[K[j]]
        if (m >= 3) { md = med(O, m); if (P[k] > md * (1 + pct / 100)) {
          x = sprintf("%.0f%% above the median %.1f A of the other %d feeds (WARN over %d%%)", (P[k] / md - 1) * 100, md, m, pct); w = w ", " x
          if (!pre) printf "HSC WARN %s, %s; WARN only, not a finding\n", h, x } }
        if (pre && w != "") printf "HSC PRE %s: %s\n", h, substr(w, 3) }
      printf "HSC peak after %s: IOUT peak A (A6h)%s; TEMP peak C (AFh)%s\n", st, s, c }
  ' "$2" "$([ -s "${3:-}" ] && echo "$3" || echo /dev/null)"
}
hsc_poll_start() {  # <file>: READ_VIN of every HSC, every HSC_VIN_POLL_S, until hsc_poll_stop (our own subshell, by pid) or run.sh ends
  local main=$$
  ( trap - USR1 INT TERM EXIT; local t a v i   # the loop ends itself with run.sh too (an exit that skips hsc_poll_stop)
    while kill -0 "$main" 2> /dev/null; do for t in $HSC_TRAYS; do for a in $HSC_ADDRS; do
      # VIN and IOUT are two ipmitool calls, read back to back: they are not one atomic sample
      v=$(hsc_rd "$t" "$a" 0x88); [ -n "$v" ] && { i=$(hsc_rd "$t" "$a" 0x8C); printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$t" "$a" "$v" "${i:--}"; }; done; done
      sleep "${HSC_VIN_POLL_S:-2}"; done ) < /dev/null >> "$1" 2> /dev/null &
  HSC_POLL_PID=$!
}
hsc_poll_stop() { [ -n "$HSC_POLL_PID" ] || return 0; kill "$HSC_POLL_PID" 2> /dev/null; wait "$HSC_POLL_PID" 2> /dev/null; HSC_POLL_PID=""; }
fl_hsc_f() { local d=$FL_OUT/hsc; mkdir -p "$d"; echo "$d/$(echo "$1" | tr -c 'A-Za-z0-9\n' _)"; }
fl_hsc_say() { say "$1"; echo "$1" >> "$FL_OUT/load.txt"; }
fl_hsc_none() {  # <why>: once per run, plainly; never a finding
  [ -e "$(fl_hsc_f noroute)" ] && return 0; : > "$(fl_hsc_f noroute)"
  fl_hsc_say "HSC check not run: $1; tray 12 V undervoltage latches and min VIN not judged"
}
fl_hsc_before() {  # <phase>: the read before the leg, then the VIN sampler
  local f; HSC_LEG=""
  if [ ${#HSC_IP[@]} = 0 ] && ! hsc_route; then fl_hsc_none "no read route ($HSC_WHY)"; return 0; fi
  f=$(fl_hsc_f "$1"); rm -f "$f.before" "$f.vin" "$f.after"
  hsc_snap "$f.before"; rm -f "$f.peak0" "$f.peak"
  if ! awk -F'\t' '$4 != "-" || $5 != "-" { f = 1 } END { exit !f }' "$f.before"; then
    fl_hsc_none "no HSC answered (${HSC_IP[*]} raw 0x06 0x52, before $(fl_pn "$1"))"; return 0; fi
  hsc_oclim
  hsc_peaks "$f.peak0"   # just before the leg: zeroes the read-clear peaks, so the read after it is this leg's peak
  hsc_step_line before "$(fl_pn "$1")" "$f.before" "$f.peak0"
  HSC_LEG=$1; : > "$f.vin"; hsc_poll_start "$f.vin"
}
fl_hsc_after() {  # <phase>: stop the sampler, read again, judge each HSC
  local f l k
  [ "$HSC_LEG" = "$1" ] || return 0
  hsc_poll_stop; HSC_LEG=""; f=$(fl_hsc_f "$1")
  hsc_peaks "$f.peak"   # right after the leg: this leg's peak current and temperature per feed
  hsc_snap "$f.after"
  hsc_step_line after "$(fl_pn "$1")" "$f.after" "$f.peak"
  while IFS= read -r l; do case $l in
    "HSC WARN "*) fl_find "$l" ;;   # sag or VIN floor: a warn, never a FAIL
    "HSC FAIL "*) fl_find "$l"   # the end read names this HSC once only (fl_hsc_latch_end)
      [[ $l =~ ^HSC\ FAIL\ tray\ ([1-4])\ .*\ HSC\ (0x4[02]): ]] && fl_pn "$1" > "$(fl_hsc_f "failed_${BASH_REMATCH[1]}_${BASH_REMATCH[2]}")" ;;
    "NOTE "*) k=${l#NOTE }; l=${k#* }; k=$(fl_hsc_f "noted_${k%% *}")
      [ -e "$k" ] || { : > "$k"; fl_hsc_say "$l"; } ;;
    ?*) fl_hsc_say "$l" ;;
  esac; done < <(hsc_judge "$(fl_pn "$1")" "$f.before" "$f.after" "$f.vin"; hsc_peak_judge "$(fl_pn "$1")" "$f.peak" "$(fl_hsc_f oc_limits)")
  if [ "$1" = busbar ]; then while IFS= read -r l; do fl_hsc_say "$l"; done < <(hsc_load_sum "$(fl_pn "$1")" "$f.vin"); fi   # the bus-bar stage: its load summary
  return 0
}
hsc_judge() {  # <stage name> <before> <after> <vin samples>: "HSC FAIL ...", "HSC WARN ...", "NOTE <tray_addr> <line>", evidence and one summary line
  awk -F'\t' -v st="$1" -v lim="${HSC_VIN_MIN_V:-11.0}" -v rmo="${HSC_SAG_MOHM:-7.5}" -v ia="${HSC_SAG_MIN_A:-40}" -v r1="01:00.0-08:00.0" -v r2="41:00.0-48:00.0" \
      -v r3="c1:00.0-c8:00.0" -v r4="81:00.0-88:00.0" -v fb="$2" -v fa="$3" '
    function v(x) { return x * 0.03125 }
    function uv(x) { return int(x / 16) % 4 }   # STATUS_INPUT bit 5 VIN_UV_WARN, bit 4 VIN_UV_FAULT (mawk: no and())
    function bits(x,  s) { s = ""; if (int(x / 32) % 2) s = "VIN_UV_WARN"; if (int(x / 16) % 2) s = s (s ? "+" : "") "VIN_UV_FAULT"; return s }
    function wb(x, b) { return x != "-" && int(x / b) % 2 }   # one bit: STATUS_INPUT VIN_UV_WARN 32, VIN_UV_FAULT 16; STATUS_WORD OFF 64, IOUT_OC 16
    function nb(o, x, b) { return o != "-" && wb(x, b) && !wb(o, b) }   # clear before, set after, each bit on its own
    function hx(x, w) { return x == "-" ? "not read" : sprintf("0x%0" w "x", x) }
    function med(A, n,  i, j, x) { for (i = 2; i <= n; i++) { x = A[i]; for (j = i - 1; j >= 1 && A[j] > x; j--) A[j + 1] = A[j]; A[j + 1] = x }
      return n % 2 ? A[(n + 1) / 2] : (A[n / 2] + A[n / 2 + 1]) / 2 }
    function lo(k, x) { if (x != "-" && (!(k in mn) || v(x) < mn[k])) mn[k] = v(x); if (x != "-" && (!(k in pk) || v(x) > pk[k])) pk[k] = v(x) }
    function good(x, y) { return x != 65535 && y != 65535 && v(x) >= 6 && v(x) <= 15 && y * 0.0625 <= 250 }
    function p10(A, n,  i, j, x) { for (i = 2; i <= n; i++) { x = A[i]; for (j = i - 1; j >= 1 && A[j] > x; j--) A[j + 1] = A[j]; A[j + 1] = x }
      j = int((n + 9) / 10); return A[j < 1 ? 1 : j] }   # nearest rank: the ceil(n/10)-th smallest
    BEGIN { R[1] = r1; R[2] = r2; R[3] = r3; R[4] = r4 }
    FILENAME == fb { k = $1 "\t" $2; K[++n] = k; bsw[k] = $3; bsi[k] = $4; bv[k] = $5; lo(k, $5); lo(k, $6); lo(k, $7); bmn[k] = $6; next }
    FILENAME == fa { k = $1 "\t" $2; asw[k] = $3; asi[k] = $4; lo(k, $5); lo(k, $6); lo(k, $7); next }
    # a sample with VIN or IOUT reading 0xFFFF or out of range (VIN outside 6-15 V, IOUT over 250 A) feeds no sag number
    { k = $2 "\t" $3; lo(k, $4); ns[k]++; if ($5 != "" && $5 != "-" && good($4, $5)) { m = ++np[k]; SV[k, m] = v($4); SI[k, m] = $5 * 0.0625 } }
    END {
      for (i = 1; i <= n; i++) { k = K[i]; split(k, p, "\t"); t = p[1]; a = p[2]; why = ""; pre = ""; sw = ""
        if (bsi[k] != "-" && uv(bsi[k])) pre = bits(bsi[k]) " set (STATUS_INPUT " hx(bsi[k], 2) ")"
        bl = ""; if (bv[k] != "-") bl = v(bv[k]); if (bmn[k] != "-" && (bl == "" || v(bmn[k]) < bl)) bl = v(bmn[k])
        vlow = bl != "" && bl < lim
        if (vlow) pre = pre (pre ? ", " : "") sprintf("min VIN %.2f V", bl)
        if (pre) printf "NOTE %s_%s HSC tray %d (%s) HSC %s latched before load: a new dip on this tray cannot be seen until a power cycle clears it (%s before %s)\n", t, a, t, R[t], a, pre, st
        nu = 32 * nb(bsi[k], asi[k], 32) + 16 * nb(bsi[k], asi[k], 16)
        if (nu) why = bits(nu) " latched during " st " (clear before; STATUS_INPUT " hx(bsi[k], 2) " -> " hx(asi[k], 2) ", STATUS_WORD " hx(bsw[k], 4) " -> " hx(asw[k], 4) ")"
        if (nb(bsw[k], asw[k], 64)) sw = "OFF"
        if (nb(bsw[k], asw[k], 16)) sw = sw (sw ? "+" : "") "IOUT_OC"
        if (sw) why = why (why ? "; " : "") sw " set during " st " (STATUS_WORD " hx(bsw[k], 4) " -> " hx(asw[k], 4) ")"
        if (why) printf "HSC FAIL tray %d (%s) HSC %s: %s\n", t, R[t], a, why
        if (!vlow && (k in mn) && mn[k] < lim) printf "HSC WARN tray %d (%s) HSC %s: min VIN %.2f V during %s, below %.1f V (%d sample(s)); WARN only, not a finding\n", t, R[t], a, mn[k], st, lim, ns[k] + 0
        # 12 V sag: R = (idle VIN - VIN) / IOUT at IOUT >= ia; idle = median VIN at 2 A < IOUT <= 30 A, else the read before
        ni = 0; for (j = 1; j <= np[k]; j++) if (SI[k, j] > 2 && SI[k, j] <= 30) Q[++ni] = SV[k, j]
        idl = ""; isrc = ""; if (ni >= 3) { idl = med(Q, ni); isrc = ni " idle samples" } else if (bv[k] != "-") { idl = v(bv[k]); isrc = "read before the leg" }
        nr = 0; rx = ""; if (idl != "") for (j = 1; j <= np[k]; j++) if (SI[k, j] >= ia) { r = (idl - SV[k, j]) / SI[k, j] * 1000; RR[++nr] = r
          if (rx == "" || r > rx) { rx = r; xv = SV[k, j]; xi = SI[k, j] } }
        if (nr) { g = g sprintf(" t%d/%s %.1f", t, a, rx)
          if (rx > rmo) printf "HSC WARN tray %d (%s) HSC %s: 12 V sag %.1f mOhm during %s, over %.1f mOhm (VIN %.2f V at IOUT %.1f A, idle VIN %.2f V from %s; median %.1f mOhm over %d sample(s) at IOUT >= %d A); WARN only, not a finding\n", t, R[t], a, rx, st, rmo, xv, xi, idl, isrc, med(RR, nr), nr, ia }
        # per feed: VIN p10 and median and IOUT median of the samples at IOUT >= ia, and the effective
        # resistance (idle VIN - VIN median) / IOUT median; no number for an HSC that read OFF or had no good sample at load
        nl = 0; for (j = 1; j <= np[k]; j++) if (SI[k, j] >= ia) { LV[++nl] = SV[k, j]; LI[nl] = SI[k, j] }
        if (wb(bsw[k], 64) || wb(asw[k], 64)) FW[i] = "OFF (STATUS_WORD " hx(bsw[k], 4) " -> " hx(asw[k], 4) ")"
        else if (!nl) FW[i] = (np[k] ? "no good sample at IOUT >= " ia " A" : "no good VIN and IOUT sample")
        else if (idl == "") FW[i] = "no idle VIN"
        else { FM[i] = med(LV, nl); FP[i] = p10(LV, nl); FI[i] = med(LI, nl); FN[i] = nl; FSC[i] = isrc; FD[i] = idl
          if (FI[i] > 0) { FR[i] = (idl - FM[i]) / FI[i] * 1000; HR[i] = 1 } else FW[i] = "IOUT median 0 A" }
        FT[i] = t; FA[i] = a
        s = s sprintf(" t%d/%s %s", t, a, (k in mn) ? sprintf("%.2f-%.2f", mn[k], pk[k]) : "not read")
        if (asi[k] != "-" && uv(asi[k])) u = u " t" t "/" a ":" bits(asi[k]) }
      for (i = 1; i <= n; i++) { t = FT[i]; a = FA[i]
        if (!(i in HR)) { printf "HSC feed tray %d (%s) HSC %s during %s: no number (%s)\n", t, R[t], a, st, FW[i] (i in FM ? sprintf("; VIN p10 %.2f V, median %.2f V over %d sample(s) at IOUT >= %d A", FP[i], FM[i], FN[i], ia) : ""); continue }
        no = 0; for (j = 1; j <= n; j++) if (j != i && (j in HR)) O[++no] = FR[j]
        gp = no ? sprintf("gap %+.1f mOhm to %.1f mOhm, the median of the other %d feed(s)", FR[i] - med(O, no), med(O, no), no) : "gap: no other feed with a number"
        printf "HSC feed tray %d (%s) HSC %s during %s: VIN p10 %.2f V, median %.2f V over %d sample(s) at IOUT >= %d A; IOUT median %.1f A; R %.1f mOhm (idle VIN %.2f V from %s); %s\n", t, R[t], a, st, FP[i], FM[i], FN[i], ia, FI[i], FR[i], FD[i], FSC[i], gp }
      printf "HSC after %s: VIN min-peak V%s; UV latched: %s\n", st, s, u ? substr(u, 2) : "none"
      if (g) printf "HSC sag after %s: R max mOhm at IOUT >= %d A%s (WARN over %.1f)\n", st, ia, g, rmo }
  ' "$2" "$3" "$4"
}
# Run-level latch record: before C2, after the one CLEAR_FAULTS, after C9.
hsc_load_sum() {  # <stage name> <vin samples>: one line per tray, its HSC IOUT mean and p90 (the tray's HSCs summed per poll round) and
  # its lowest VIN over the leg (the bus-bar stage with its max-util part); reads the sampler's file only
  awk -F'\t' -v st="$1" -v r1="01:00.0-08:00.0" -v r2="41:00.0-48:00.0" -v r3="c1:00.0-c8:00.0" -v r4="81:00.0-88:00.0" '
    function p90(A, n,  i, j, x) { for (i = 2; i <= n; i++) { x = A[i]; for (j = i - 1; j >= 1 && A[j] > x; j--) A[j + 1] = A[j]; A[j + 1] = x }
      i = int(0.9 * n); if (i < 0.9 * n) i++; return A[i] }
    NF >= 4 && $2 ~ /^[1-4]$/ { t = $2; k = t SUBSEP $3; if (!(t in mv) || $4 * 0.03125 < mv[t]) mv[t] = $4 * 0.03125
      if (!((t, $3) in seen)) { seen[t, $3] = 1; H[t] = H[t] " " $3 }
      if ($5 != "" && $5 != "-") { m = ++c[k]; I[k, m] = $5 * 0.0625 } }
    END { R[1] = r1; R[2] = r2; R[3] = r3; R[4] = r4
      for (t = 1; t <= 4; t++) { if (!(t in mv)) { printf "HSC load tray %d (%s) during %s: not read\n", t, R[t], st; continue }
        nh = split(substr(H[t], 2), A, " "); n = 0; first = 1
        for (h = 1; h <= nh; h++) { m = c[t, A[h]] + 0; if (first || m < n) n = m; first = 0 }
        s = 0; for (j = 1; j <= n; j++) { S[j] = 0; for (h = 1; h <= nh; h++) S[j] += I[t, A[h], j]; s += S[j] }
        if (n) printf "HSC load tray %d (%s) during %s: IOUT mean %.1f A, p90 %.1f A (%d poll(s), HSC %s summed); lowest VIN %.2f V\n", t, R[t], st, s / n, p90(S, n), n, substr(H[t], 2), mv[t]
        else printf "HSC load tray %d (%s) during %s: IOUT not read; lowest VIN %.2f V\n", t, R[t], st, mv[t] }
    }' "$2"
}
hsc_isset() {  # <STATUS_WORD> <STATUS_INPUT> <bit>: 0 when that bit is set ("-" or empty: not read, never set)
  local w=${1:--} i=${2:--}
  case $3 in VIN_UV_WARN) [ "$i" != - ] && (( i & 0x20 )) ;; VIN_UV_FAULT) [ "$i" != - ] && (( i & 0x10 )) ;;
    OFF) [ "$w" != - ] && (( w & 0x40 )) ;; IOUT_OC) [ "$w" != - ] && (( w & 0x10 )) ;; CML) [ "$w" != - ] && (( w & 0x02 )) ;; *) false ;; esac
}
hsc_names() {  # <STATUS_WORD> <STATUS_INPUT>: "VIN_UV_WARN+OFF", "CML", "none", "+unknown" when one of the two did not read, or "not read"
  local b s=""; [ "${1:--}${2:--}" = "--" ] && { echo "not read"; return 0; }
  for b in VIN_UV_WARN VIN_UV_FAULT OFF IOUT_OC CML; do hsc_isset "$1" "$2" $b && s+="+$b"; done
  { [ "${1:--}" = - ] || [ "${2:--}" = - ]; } && s+="+unknown"
  echo "${s:-+none}" | cut -c2-
}
hsc_pw() { local s="+$1+"; s=${s//+CML+/+}; s=${s//+unknown+/+}; s=${s#+}; s=${s%+}; echo "${s:-none}"; }   # <names>: the power bits only
hsc_cmltxt() {  # <STATUS_CML 0xNN or ->: "comm fault (CML, STATUS_CML 0x80: invalid or unsupported command), not power"
  local c=${1:--} d="" ; [ "$c" = - ] && { echo "comm fault (CML, STATUS_CML not read), not power"; return 0; }
  (( c & 0x80 )) && d+=", invalid or unsupported command"; (( c & 0x40 )) && d+=", invalid data"; (( c & 0x20 )) && d+=", PEC failed"
  (( c & 0x10 )) && d+=", memory fault"; (( c & 0x08 )) && d+=", processor fault"; (( c & 0x02 )) && d+=", other comm fault"
  (( c & 0x01 )) && d+=", other memory or logic fault"
  echo "comm fault (CML, STATUS_CML $c${d:+: ${d#, }}), not power"
}
hsc_lsnap() {  # <file>: STATUS_WORD, STATUS_INPUT (and STATUS_CML where CML is set) of every tray HSC (tsv, a header first; "-" = not read)
  local t a sw si cm fr fl so vo   # plus FAULT_RECORD, FAULT_RECORD_LAST, STATUS_IOUT and READ_VOUT V
  printf '# tray\thsc\tbdfs\tstatus_word\tstatus_input\tset\tstatus_cml\tfault_record\tfault_record_last\tstatus_iout\tvout\n' > "$1"
  for t in $HSC_TRAYS; do for a in $HSC_ADDRS; do
    sw=$(hsc_rd "$t" "$a" 0x79); si=$(hsc_rd "$t" "$a" 0x7C); cm=""
    [ -n "$sw" ] && (( sw & 0x02 )) && cm=$(hsc_rd "$t" "$a" 0x7E)
    fr=$(hsc_rd "$t" "$a" 0xFA); fl=$(hsc_rd "$t" "$a" 0xFB); so=$(hsc_rd "$t" "$a" 0x7B); vo=$(hsc_rd "$t" "$a" 0x8B)
    [ -n "$sw" ] && sw=$(printf '0x%04x' "$sw"); [ -n "$si" ] && si=$(printf '0x%02x' "$si"); [ -n "$cm" ] && cm=$(printf '0x%02x' "$cm")
    [ -n "$fr" ] && fr=$(printf '0x%04x' "$fr"); [ -n "$fl" ] && fl=$(printf '0x%04x' "$fl"); [ -n "$so" ] && so=$(printf '0x%02x' "$so")
    [ -n "$vo" ] && vo=$(awk -v r="$vo" 'BEGIN { printf "%.2f", r * 0.03125 }')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$t" "$a" "$(hsc_bdfs "$t")" "${sw:--}" "${si:--}" "$(hsc_names "${sw:--}" "${si:--}")" "${cm:--}" \
      "${fr:--}" "${fl:--}" "${so:--}" "${vo:--}" >> "$1"
  done; done
}
hsc_lget() { [ -s "$1" ] && awk -F'\t' -v t="$2" -v a="$3" -v c="$4" '!/^#/ && $1 == t && $2 == a { print $c }' "$1"; return 0; }   # <file> <tray> <addr> <col>
hsc_lcell() { [ -s "$1" ] || { echo "not read"; return 0; }; local w s c; w=$(hsc_lget "$1" "$2" "$3" 4); s=$(hsc_lget "$1" "$2" "$3" 5); c=$(hsc_lget "$1" "$2" "$3" 7)
  [ "${w:--}${s:--}" = "--" ] && { echo "not read"; return 0; }; echo "$(hsc_names "$w" "$s") (${w:--}/${s:--}$([ "${c:--}" != - ] && echo "/7Eh $c"))"; }
hsc_unk() {  # <sw> <si> <when>: why the HSC's bits cannot all be judged (one register read, the other failed or NACKed)
  [ "${1:--}${2:--}" = "--" ] && return 0
  [ "${1:--}" = - ] && printf '; STATUS_WORD not read %s (failed or NACKed): OFF, IOUT_OC and CML unknown' "$3"
  [ "${2:--}" = - ] && printf '; STATUS_INPUT not read %s (failed or NACKed): VIN_UV_WARN and VIN_UV_FAULT unknown' "$3"
  return 0
}
hsc_latch_judge() {  # <before> <after> [<cleared>]: per HSC "tray\taddr\tbdfs\tverdict\twhy", power bits graded per bit from the reads;
  # CML apart: a comm fault, WARN at most, never FAIL, never "ok"; a register that did not read: "unknown", never "ok"
  local t a bd sw si x cm esw esi ecm csw csi cl b new pre kept ctx cmw unk ung usw usi rb rc nrd
  while IFS=$'\t' read -r t a bd sw si x cm _; do
    case $t in '#'*|'') continue ;; esac
    esw=$(hsc_lget "$2" "$t" "$a" 4); esi=$(hsc_lget "$2" "$t" "$a" 5); ecm=$(hsc_lget "$2" "$t" "$a" 7)
    csw=""; csi=""; cl=""; [ -n "${3:-}" ] && { csw=$(hsc_lget "$3" "$t" "$a" 4); csi=$(hsc_lget "$3" "$t" "$a" 5); }
    [ "${csw:--}${csi:--}" != "--" ] && [ -e "$(fl_hsc_f "cleared_${t}_${a}")" ] && cl=1
    new=""; pre=""; kept=""; usw=""; usi=""; nrd=""
    for b in VIN_UV_WARN VIN_UV_FAULT OFF IOUT_OC; do   # the power bits: CML is never one of them
      if hsc_isset "$sw" "$si" $b; then
        # set before: new only when its own after-clear register (STATUS_WORD for OFF/IOUT_OC, STATUS_INPUT for UV) was read
        # clear and it is set again at the end; that register not read after the clear: pre-existing
        case $b in OFF|IOUT_OC) rc=$csw; rb=STATUS_WORD ;; *) rc=$csi; rb=STATUS_INPUT ;; esac
        if [ -n "$cl" ] && [ "${rc:--}" != - ] && ! hsc_isset "$csw" "$csi" $b && hsc_isset "$esw" "$esi" $b; then new+="+$b"; else pre+="+$b"; fi
        [ -n "$cl" ] && hsc_isset "$csw" "$csi" $b && kept+="+$b"
        [ -n "$cl" ] && [ "${rc:--}" = - ] && nrd+="; $b: after-clear $rb not read"
      elif hsc_isset "$esw" "$esi" $b; then
        # set at the end only: new when its own before-register was read, or CLEAR_FAULTS was sent and the after-clear read shows it clear
        case $b in OFF|IOUT_OC) rb=$sw; rc=$csw ;; *) rb=$si; rc=$csi ;; esac
        if [ "${rb:--}" != - ] || { [ -n "$cl" ] && [ "${rc:--}" != - ] && ! hsc_isset "$csw" "$csi" $b; }; then new+="+$b"
        else case $b in OFF|IOUT_OC) usw+="+$b" ;; *) usi+="+$b" ;; esac; fi
      fi
    done
    cmw=""
    if hsc_isset "$sw" - CML; then
      if [ -n "$cl" ] && ! hsc_isset "$csw" - CML && hsc_isset "$esw" - CML; then cmw="$(hsc_cmltxt "${ecm:--}"): set during the run (cleared by CLEAR_FAULTS before C2)"
      else cmw="$(hsc_cmltxt "${cm:--}"): latched before the run, pre-existing$([ -n "$cl" ] && ! hsc_isset "$csw" - CML && echo ", cleared by CLEAR_FAULTS")"; fi
    elif hsc_isset "$esw" - CML; then cmw="$(hsc_cmltxt "${ecm:--}"): set during the run"; fi
    unk="$(hsc_unk "$sw" "$si" "before C2")$(hsc_unk "$esw" "$esi" "after C9")"
    if [ -n "$cl" ]; then ctx="clear after CLEAR_FAULTS before C2"; else ctx="clear before the run, no CLEAR_FAULTS"; fi
    if [ -z "$cl" ] && [ -e "$(fl_hsc_f "cleared_${t}_${a}")" ]; then rc="CLEAR_FAULTS sent, no read after it"
    elif [ -n "$cl" ]; then rc="not clear after CLEAR_FAULTS"; else rc="no CLEAR_FAULTS"; fi
    ung="${usw:+; STATUS_WORD not read before C2, $rc: ${usw#+} at the end not graded}${usi:+; STATUS_INPUT not read before C2, $rc: ${usi#+} at the end not graded}"
    if [ -n "$new" ]; then printf '%s\t%s\t%s\tFAIL\t%s set during the run (%s)%s%s%s%s\n' "$t" "$a" "$bd" "${new#+}" "$ctx" "${pre:+; ${pre#+} pre-existing}$nrd" "${cmw:+; $cmw}" "$ung" "$unk"
    elif [ -n "$pre" ]; then printf '%s\t%s\t%s\tWARN\t%s latched before the run, pre-existing%s%s%s%s\n' "$t" "$a" "$bd" "${pre#+}" \
      "$(if [ -n "$kept" ]; then echo "; ${kept#+} still set after CLEAR_FAULTS$nrd"; elif [ -n "$nrd" ]; then echo "$nrd"; elif [ -n "$cl" ]; then echo "; cleared by CLEAR_FAULTS"
         elif [ -e "$(fl_hsc_f "cleared_${t}_${a}")" ]; then echo "; CLEAR_FAULTS sent, no re-read"; fi)" "${cmw:+; $cmw}" "$ung" "$unk"
    elif [ -n "$ung" ]; then printf '%s\t%s\t%s\tnot graded\t%s%s\n' "$t" "$a" "$bd" "${ung#; }" "${cmw:+; $cmw}"
    elif [ "${sw:--}${si:--}" = "--" ]; then printf '%s\t%s\t%s\tnot read\tno read before C2\n' "$t" "$a" "$bd"
    elif [ "${esw:--}${esi:--}" = "--" ]; then printf '%s\t%s\t%s\tnot read\tno read after C9%s\n' "$t" "$a" "$bd" "${cmw:+; $cmw}"
    elif [ -n "$unk" ]; then printf '%s\t%s\t%s\tunknown\t%s%s\n' "$t" "$a" "$bd" "${unk#; }" "${cmw:+; $cmw}"
    elif [ -n "$cmw" ]; then printf '%s\t%s\t%s\tWARN\t%s\n' "$t" "$a" "$bd" "$cmw"
    else printf '%s\t%s\t%s\tok\t\n' "$t" "$a" "$bd"; fi
  done < "$1"
}
fl_hsc_latch_start() {  # before C2: record every HSC (each bit already set: a pre-existing WARN), then CLEAR_FAULTS once to each unless one reads OFF
  local b=$FL_OUT/hsc_latch_before.tsv c=$FL_OUT/hsc_latch_cleared.tsv tb t a bd sw si x lim off="" n=0 y cm fr fl so vo
  tb=$(fl_hsc_f latch_table); rm -f "$tb" "$(fl_hsc_f latch_state)" "$(fl_hsc_f notrun)" "$c" "$FL_OUT/hsc_latch_after.tsv" "$(fl_hsc_f oc_limits)"
  if [ ${#HSC_IP[@]} = 0 ] && ! hsc_route; then
    echo "${HSC_NR:-$HSC_WHY}" > "$(fl_hsc_f notrun)"
    echo "HSC latch: not read ($HSC_WHY); no CLEAR_FAULTS sent" > "$tb"; fl_hsc_say "$(cat "$tb")"; return 0; fi
  hsc_lsnap "$b"
  if ! awk -F'\t' '!/^#/ && ($4 != "-" || $5 != "-") { f = 1 } END { exit !f }' "$b"; then
    hsc_noanswer > "$(fl_hsc_f notrun)"
    echo "HSC latch: not read (no HSC answered ${HSC_IP[*]} raw 0x06 0x52); no CLEAR_FAULTS sent" > "$tb"; fl_hsc_say "$(cat "$tb")"; return 0; fi
  lim=$(for t in $HSC_TRAYS; do for a in $HSC_ADDRS; do x=$(hsc_rd "$t" "$a" 0x58); [ -n "$x" ] && awk -v r="$x" 'BEGIN { printf "%.2f V (58h raw %d)\n", (r % 128) * 0.125, r % 128 }'; done; done | sort | uniq -c | awk '{ c = $1; sub(/^ *[0-9]+ /, ""); printf "%s%s on %d HSC(s)", (n++ ? ", " : ""), $0, c }')
  echo "HSC UV warn limit (VIN_UV_WARN_LIMIT): ${lim:-not read}" > "$(fl_hsc_f latch_limit)"
  local pre=""   # every bit already set: one HSC WARN line for all HSCs, never a verdict
  while IFS=$'\t' read -r t a bd sw si x cm _; do case $t in '#'*) continue ;; esac
    x=$(hsc_pw "$x"); case $x in none|"not read") ;; *) pre="$pre; tray $t ($bd) HSC $a: $x (STATUS_WORD $sw, STATUS_INPUT $si)" ;; esac
    hsc_isset "$sw" - CML && pre="$pre; tray $t ($bd) HSC $a: $(hsc_cmltxt "${cm:--}") (STATUS_WORD $sw)"
  done < "$b"
  # gate (c): every HSC's STATUS_WORD/STATUS_INPUT in the run output before any clear
  fl_hsc_say "HSC latch before C2 (STATUS_WORD/STATUS_INPUT): $(awk -F'\t' '!/^#/ { s = s ", t" $1 "/" $2 " " $4 "/" $5 " " $6 } END { print substr(s, 3) }' "$b")"
  # FAULT_RECORD, STATUS_IOUT, VOUT and the peaks since the last read, logged before any clear
  x=""; while IFS=$'\t' read -r t a bd sw si y cm fr fl so vo; do case $t in '#'*) continue ;; esac
    x+=", t$t/$a $(hsc_frdec "$fr")/$(hsc_frdec "$fl") iout $so vout $vo"; done < "$b"
  fl_hsc_say "HSC fault record before C2 (FAULT_RECORD/FAULT_RECORD_LAST, STATUS_IOUT, VOUT V; FAULT_RECORD_LAST is the last power-off, logged only): ${x#, }"
  hsc_peaks "$(fl_hsc_f peak_run_start)"; hsc_oclim
  # the peaks from before the run join the latch items: one HSC WARN host line, never per feed, never a verdict
  while IFS= read -r x; do case $x in "HSC PRE "*) pre="$pre; ${x#HSC PRE }" ;; *) fl_hsc_say "$x" ;; esac
  done < <(hsc_peak_judge "the time before this run (since the last peak read)" "$(fl_hsc_f peak_run_start)" "$(fl_hsc_f oc_limits)" pre)
  [ -n "$pre" ] && fl_find "HSC WARN host: ${pre#; }; latched before the run: pre-existing, not from this run"
  off=$(awk -F'\t' '!/^#/ && $6 ~ /(^|\+)OFF(\+|$)/ { s = s ", tray " $1 " (" $3 ") HSC " $2 } END { print substr(s, 3) }' "$b")
  if [ -n "$off" ]; then   # gate (b): one HSC OFF -> no clear to any HSC this run
    echo kept > "$(fl_hsc_f latch_state)"; fl_hsc_say "HSC latch clear skipped: HSC OFF ($off); no CLEAR_FAULTS sent to any HSC this run"; return 0; fi
  HSC_CLEAR_OK=1
  while IFS=$'\t' read -r t a bd sw si x _; do case $t in '#'*) continue ;; esac
    if [ "$sw" = - ]; then   # OFF cannot be judged without STATUS_WORD, so no clear to that HSC
      fl_hsc_say "HSC latch clear skipped tray $t ($bd) HSC $a: STATUS_WORD not read, OFF cannot be judged; no CLEAR_FAULTS sent to it"; continue; fi
    hsc_clear_faults "$t" "$a" && n=$((n + 1)) || fl_hsc_say "HSC latch: CLEAR_FAULTS failed tray $t ($bd) HSC $a (rc $?)"
  done < "$b"
  HSC_CLEAR_OK=0; hsc_lsnap "$c"; echo cleared > "$(fl_hsc_f latch_state)"
  fl_hsc_say "HSC latch: CLEAR_FAULTS (03h) sent once to $n HSC(s) before C2; after it: $(awk -F'\t' '!/^#/ && $6 != "none" { s = s " t" $1 "/" $2 ":" $6 } END { print s ? substr(s, 2) : "all clear" }' "$c")"
}
fl_hsc_latch_end() {  # after C9: read again; a bit set during the run is an HSC FAIL unless a leg already named that HSC
  local st t a bd v why; st=$(cat "$(fl_hsc_f latch_state)" 2> /dev/null); [ -n "$st" ] || return 0
  hsc_lsnap "$FL_OUT/hsc_latch_after.tsv"
  hsc_latch_judge "$FL_OUT/hsc_latch_before.tsv" "$FL_OUT/hsc_latch_after.tsv" "$FL_OUT/hsc_latch_cleared.tsv" > "$(fl_hsc_f latch_judged)"
  while IFS=$'\t' read -r t a bd v why; do
    [ "$v" = FAIL ] && [ ! -e "$(fl_hsc_f "failed_${t}_${a}")" ] && fl_find "HSC FAIL tray $t ($bd) HSC $a: $why"
    [ "$v" = WARN ] && [[ $why == "comm fault (CML"*": set during the run"* ]] && fl_find "HSC WARN tray $t ($bd) HSC $a: $why"
  done < "$(fl_hsc_f latch_judged)"
  return 0
}
hsc_noanswer() {  # why no HSC answered: no in-band IPMI device node (an in-band route only), else the plain no-answer
  local d; [[ " ${HSC_IP[*]} " == *" -H "* ]] && { echo "no HSC answered"; return 0; }   # out of band: no local device node
  for d in ${HSC_IPMI_DEVS:-/dev/ipmi0 /dev/ipmi/0 /dev/ipmidev/0}; do [ -e "$d" ] && { echo "no HSC answered"; return 0; }; done
  echo "no ipmi device (${HSC_IPMI_DEVS:-/dev/ipmi0 /dev/ipmi/0 /dev/ipmidev/0} missing)"
}
fl_hsc_latch_table() {  # in fl_report: the before/after table per tray and HSC (or the one not-read line)
  local tb st t a bd v why f; tb=$(fl_hsc_f latch_table); st=$(cat "$(fl_hsc_f latch_state)" 2> /dev/null)
  if [ -z "$st" ]; then   # the end summary never hides an HSC check that did not run, in every mode
    echo "WARN HSC check NOT RUN: $(cat "$(fl_hsc_f notrun)" 2> /dev/null || echo "not reached (the run ended before the HSC read before C2)")"
    [ -s "$tb" ] && cat "$tb"
    # the tray table names the skipped under-voltage read in plain words; the verdict stays
    # PASS or FAIL for the stages that ran (no HSC route never makes a run INCOMPLETE; the WARN is never silent)
    echo "WARN tray 12 V under-voltage read NOT RUN on trays 1-4 (VIN_UV latches and min VIN not judged); PASS/FAIL covers the stages that ran"
    return 0; fi
  [ -e "$FL_OUT/hsc_latch_after.tsv" ] || hsc_latch_judge "$FL_OUT/hsc_latch_before.tsv" /dev/null "$FL_OUT/hsc_latch_cleared.tsv" > "$(fl_hsc_f latch_judged)"   # fl_end did not run
  echo "HSC latch before/after (STATUS_WORD 79h/STATUS_INPUT 7Ch, 7Eh = STATUS_CML; CML = comm fault, not power; $([ "$st" = cleared ] && echo "CLEAR_FAULTS sent before C2" || echo "latch clear skipped: HSC OFF")):"
  while IFS=$'\t' read -r t a bd v why; do
    f=$(cat "$(fl_hsc_f "failed_${t}_${a}")" 2> /dev/null)
    printf '  tray %s (%s) HSC %s: before %s%s, after C9 %s -> %s%s\n' "$t" "$bd" "$a" "$(hsc_lcell "$FL_OUT/hsc_latch_before.tsv" "$t" "$a")" \
      "$([ "$st" = cleared ] && echo ", after clear $(hsc_lcell "$FL_OUT/hsc_latch_cleared.tsv" "$t" "$a")")" \
      "$(hsc_lcell "$FL_OUT/hsc_latch_after.tsv" "$t" "$a")" "$v" "${why:+ ($why${f:+; first in $f})}"
  done < "$(fl_hsc_f latch_judged)"
  cat "$(fl_hsc_f latch_limit)" 2> /dev/null; return 0
}
