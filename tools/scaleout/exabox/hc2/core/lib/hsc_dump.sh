#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/hsc_dump.sh (a standing rule): before the first glx_reset touches a
# tray with chips missing (C2 at the start, C9 and every recovery reset after a drop: all go through rk_glx, which calls
# rk_glx_evidence, which calls hd_dump), save that tray's power state: its CPLD fault latch and both 12 V hot-swap
# controllers (MP5990 at 8-bit 0x40 and 0x42 on the tray's bus) read in-band. A glx_reset re-powers every tray and clears
# the HSC latches, so after it they can no longer say what dropped the tray. Sourced by run.sh: functions only.
#   Reads only: `ipmitool raw 0x06 0x52 <bus> <addr> <n> <code>` (master write-read) with a whitelisted read code:
#   STATUS_WORD 0x79, STATUS_VOUT 0x7A, STATUS_IOUT 0x7B, STATUS_INPUT 0x7C, STATUS_TEMPERATURE 0x7D, STATUS_MFR_SPECIFIC
#   0x80, READ_VIN 0x88, READ_VOUT 0x8B, READ_IOUT 0x8C (VIN and VOUT 31.25 mV/LSB, IOUT 62.5 mA/LSB, as the hsc_check.sh
#   readers), FAULT_RECORD 0xFA and FAULT_RECORD_LAST 0xFB (2 bytes each). The read-clear peaks
#   0xA6/0xAF are not read here: lib/hsc_check.sh reads them around each load leg. hd_rd refuses any other code: never CLEAR_FAULTS (0x03), never a CLEAR_PEAK, never a BMC write, never agera2-app.
#   Min/peak VIN and peak IOUT: the MP5990 has no such register, so the dump says so; TRAY_HSC_PEAK_REGS
#   ("VIN_MIN=0xD1 VIN_PEAK=0xD2 IOUT_PEAK=0xD3", codes 0xD0-0xDF only, 2 bytes) reads them on a part that has them.
#   CPLD: cpld_5f.py (reg 0x1E decoded: ALIVE, TRAY_OFF, CHIPS_DOWN), the reader cpld_drop already uses; reads only, no
#   CPLD index write (dbg 0x5F/0x58 'not read: needs a CPLD index write (not allowed)'), no power or reset.
#   CPLD DIAG: the plain registers of the vendor cpld-dump.sh DIAG (0x1E alive, 0x2D therm_trip, 0x19
#   PWRGD_VR) by 1-byte reads only (hd_crd); its debug-index fields are named 'not read: needs a CPLD write'. The vendor
#   cpld-dump.sh DIAG itself never runs: it writes the CPLD debug-index registers and leaves them changed.
#   Bounded: TRAY_HSC_DUMP_S (30 s) for the whole dump, each read at most TRAY_HSC_READ_S (3 s). A read past the budget,
#   with no route (no ipmitool, no root and no sudo -n for it, a Slurm node without in-band IPMI) or with no answer is
#   written as "not read: <why>", and the glx_reset runs right after in every case.
#   Each tray is dumped once per reset step (RK_STEP): later rounds of the same step follow a glx_reset that already
#   re-powered it.
# TRAY_REPRO_HSC_IPMITOOL: the ipmitool command to use as is (tests: a fake); unset: TRAY_REPRO_IPMITOOL (ipmitool) as
# root, else through sudo -n where sudo -n -l allows its raw 0x06 0x52 read.
HD_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
declare -F pg_run > /dev/null || . "$HD_DIR/pgroups.sh"
HD_IP=(); HD_WHY=""; HD_DONE=""; HD_END=0
hd_bus() { case $1 in 1) echo 0x11 ;; 2) echo 0x13 ;; 3) echo 0x0d ;; 4) echo 0x0f ;; esac; }
hd_bdfs() { case $1 in 1) echo 01:00.0-08:00.0 ;; 2) echo 41:00.0-48:00.0 ;; 3) echo c1:00.0-c8:00.0 ;; 4) echo 81:00.0-88:00.0 ;; esac; }
hd_left() { echo $(( HD_END - $(date +%s) )); }
hd_route() {  # sets HD_IP; rc 2 with the why in HD_WHY when there is no read route
  local ip=${TRAY_REPRO_IPMITOOL:-ipmitool}
  HD_IP=(); HD_WHY=""
  if [ -n "${TRAY_REPRO_HSC_IPMITOOL:-}" ]; then read -ra HD_IP <<< "$TRAY_REPRO_HSC_IPMITOOL"
    local nm=${HD_IP[0]:-${TRAY_REPRO_HSC_IPMITOOL%% *}}
    command -v "$nm" > /dev/null 2>&1 || { HD_IP=(); HD_WHY="$nm not found"; return 2; }
  elif ! command -v "$ip" > /dev/null 2>&1; then HD_WHY="$ip not found (no in-band IPMI route)"; return 2
  elif [ "$(id -u)" = 0 ]; then HD_IP=("$ip")
  elif pg_run timeout -k 2 5 sudo -n -l "$ip" raw 0x06 0x52 0x11 0x40 0x02 0x79 > /dev/null 2>&1; then HD_IP=(sudo -n "$ip")
  else HD_WHY="no root, and sudo -n -l does not allow $ip raw 0x06 0x52"; return 2; fi
}
hd_rd() {  # <tray> <addr> <code> <bytes>: the raw bytes ("b0 b1", low byte first) on stdout; nothing when not read
  local o b t
  case $3:$4 in 0x79:2|0x7A:1|0x7B:1|0x7C:1|0x7D:1|0x80:1|0x88:2|0x8B:2|0x8C:2|0xFA:2|0xFB:2|0xD[0-9A-F]:2) ;;
    *) echo "hd_rd: refused $3/$4 (not a whitelisted read)" >&2; return 2 ;; esac
  t=$(hd_left); [ "$t" -gt "${TRAY_HSC_READ_S:-3}" ] && t=${TRAY_HSC_READ_S:-3}; [ "$t" -ge 1 ] || return 0
  local lk=(); declare -F hsc_i2c > /dev/null && lk=(hsc_i2c)   # one in-band I2C read at a time where lib/hsc_check.sh is loaded
  o=$(pg_run "${lk[@]}" timeout -k 2 "$t" "${HD_IP[@]}" raw 0x06 0x52 "$(hd_bus "$1")" "$2" "0x0$4" "$3" < /dev/null 2> /dev/null) || return 0
  read -ra b <<< "$o"
  [ "${#b[@]}" = "$4" ] && [[ "${b[*]}" =~ ^[0-9a-fA-F]{2}( [0-9a-fA-F]{2})?$ ]] && echo "${b[*]}"
}
hd_bits() {  # <register name> <value>: the set bits by name
  local n=() i
  case $1 in
    STATUS_WORD) n=(NONE_ABOVE CML TEMPERATURE VIN_UV IOUT_OC VOUT_OV OFF BUSY UNKNOWN OTHER FANS POWER_GOOD# MFR INPUT IOUT/POUT VOUT) ;;
    STATUS_INPUT) n=(PIN_OP_WARN IIN_OC_WARN IIN_OC_FAULT UNIT_OFF_LOW_VIN VIN_UV_FAULT VIN_UV_WARN VIN_OV_WARN VIN_OV_FAULT) ;;
    STATUS_IOUT) n=(POUT_OP_WARN POUT_OP_FAULT POWER_LIMIT CURRENT_SHARE IOUT_UC IOUT_OC_WARN IOUT_OC_LV_FAULT IOUT_OC_FAULT) ;;
    STATUS_TEMPERATURE) n=(b0 b1 b2 b3 UT_FAULT UT_WARN OT_WARN OT_FAULT) ;;
    STATUS_VOUT) n=(POWER_ON_TRACKING TOFF_MAX TON_MAX_FAULT VOUT_MAX_WARN VOUT_UV_FAULT VOUT_UV_WARN VOUT_OV_WARN VOUT_OV_FAULT) ;;
    *) return 0 ;; esac
  for i in "${!n[@]}"; do [ $(( ($2 >> i) & 1 )) = 1 ] && printf '%s ' "${n[$i]}"; done | sed 's/ $//'
}
hd_hsc() {  # <tray> <addr>: one line per register, "hsc <addr> <NAME> <value>" or "hsc <addr> <NAME> not read: <why>"
  local r nm c n b b0 b1 v s kv w
  for r in STATUS_WORD:0x79:2 STATUS_INPUT:0x7C:1 STATUS_IOUT:0x7B:1 STATUS_TEMPERATURE:0x7D:1 STATUS_VOUT:0x7A:1 \
      STATUS_MFR_SPECIFIC:0x80:1 READ_VIN:0x88:2 READ_VOUT:0x8B:2 READ_IOUT:0x8C:2 FAULT_RECORD:0xFA:2 FAULT_RECORD_LAST:0xFB:2; do
    IFS=: read -r nm c n <<< "$r"
    if [ "$(hd_left)" -lt 1 ]; then echo "hsc $2 $nm not read: the ${TRAY_HSC_DUMP_S:-30} s dump budget is spent"; continue; fi
    b=$(hd_rd "$1" "$2" "$c" "$n")
    if [ -z "$b" ]; then echo "hsc $2 $nm not read: no answer to ${HD_IP[*]} raw 0x06 0x52 $(hd_bus "$1") $2 0x0$n $c"
      [ "$nm" = STATUS_WORD ] && { echo "hsc $2 other registers not read: the HSC did not answer STATUS_WORD"; return 0; }
      continue; fi
    read -r b0 b1 <<< "$b"; v=$(( 16#$b0 | (${b1:+16#$b1} + 0) << 8 ))
    case $nm in
      READ_VIN|READ_VOUT) echo "hsc $2 $nm $(printf '0x%04x' "$v") $(awk -v x="$v" 'BEGIN { printf "%.3f V", x * 0.03125 }')" ;;
      READ_IOUT) echo "hsc $2 $nm $(printf '0x%04x' "$v") $(awk -v x="$v" 'BEGIN { printf "%.2f A", x * 0.0625 }')" ;;
      STATUS_WORD) s=$(hd_bits "$nm" "$v"); echo "hsc $2 $nm $(printf '0x%04x' "$v")${s:+ ($s)}" ;;
      FAULT_RECORD*) echo "hsc $2 $nm $(printf '0x%04x' "$v")" ;;
      *) s=$(hd_bits "$nm" "$v"); echo "hsc $2 $nm $(printf '0x%02x' "$v")${s:+ ($s)}" ;; esac
  done
  if [ -z "${TRAY_HSC_PEAK_REGS:-}" ]; then echo "hsc $2 min/peak VIN, peak IOUT not read: the MP5990 has no such register (TRAY_HSC_PEAK_REGS unset)"; return 0; fi
  for kv in $TRAY_HSC_PEAK_REGS; do
    w=${kv#*=}; w=0x$(echo "${w#0x}" | tr a-f A-F)
    if [ "$(hd_left)" -lt 1 ]; then echo "hsc $2 ${kv%%=*} not read: the ${TRAY_HSC_DUMP_S:-30} s dump budget is spent"; continue; fi
    b=$(hd_rd "$1" "$2" "$w" 2 2> /dev/null)
    if [ -z "$b" ]; then echo "hsc $2 ${kv%%=*} ($w) not read: refused (codes 0xD0-0xDF only) or no answer"; continue; fi
    read -r b0 b1 <<< "$b"; v=$(( 16#$b0 | 16#$b1 << 8 ))
    case ${kv%%=*} in *IOUT*) s=$(awk -v x="$v" 'BEGIN { printf "%.2f A", x * 0.0625 }') ;; *) s=$(awk -v x="$v" 'BEGIN { printf "%.3f V", x * 0.03125 }') ;; esac
    echo "hsc $2 ${kv%%=*} ($w) $(printf '0x%04x' "$v") $s"
  done
}
# CPLD DIAG by reads only: the tray CPLD fields of the vendor cpld-dump.sh DIAG
# (V1.03.01, diag_ubb_BH) that are plain register reads, 0x80 on the tray's bus: reg 0x1E ASIC1-8_alive (1 = alive), 0x2D
# ASIC1-8_therm_trip (1 = tripped), 0x19 PWRGD_VR_ASIC1-8 (1 = good). hd_crd itself
# selects no debug index; those fields are only named here, as not read. hd_crd sends only `raw 0x06 0x52 <tray bus> 0x80 0x01 <0x19|0x1E|0x2D>`.
HD_DIAG_RD="0x1E:ASIC_alive:1 0x2D:ASIC_therm_trip:0 0x19:PWRGD_VR_ASIC:1"   # reg:field:the vendor's good bit value
HD_DIAG_DBG="0xD6:oFPGA_VR_ASIC1-8_PWRON 0x5D:P1V8_VDDIO_PGOOD_A1-8 0xD5:oRST_ASIC1-8_PBRST_N 0xDD:oSPI_FPGA_ASIC1-8_RST_N
  0x58:UBB_PWR_EN/PWR_BRK_N/PE_RST_N/FORCE_POWER_ON_N/PSU_OC_WARN_ALERT_N/PWRGD_VR_P3V3
  0x59:PWRGD_VR_P1V8_CLK_GEN/P0V75_DVDD/P0V75_VDD/P0V9_PHY/P1V5_PHY/P1V8_PHY
  0xD0:oUBB_PWR_READY_N/oFPGA_VR_P3V3_EN/P1V8_CLK_GEN_EN/P1V8_PHY_EN
  0xD1:oFPGA_VR_P0V75_DVDD_EN/P0V75_VDD_EN/P0V9_PHY_EN/P1V5_PHY_EN/oHSC_P12V_A_EN/oHSC_P12V_B_EN
  0x5F:PWRGD_P12V_A_HS/PWRGD_P12V_B_HS 0xD9:oRST_UBB_RT1_N 0xDB:oI2C_BUF_TMP75_EN 0xDE:oSGPIO_OE_N/oSGPIO_RST_N
  0xD2:oASIC1-8_GPIO53_AUX_POWER_ALERT 0xD3:oASIC1-8_GPIO54_PSU_OC_WARN_ALERT"
hd_crd() {  # <tray> <reg>: the tray CPLD register's byte (2 hex digits) on stdout; nothing when not read; rc 2 when refused
  local o b t
  [ $# = 2 ] || { echo "hd_crd: refused $*: takes <tray> <reg> only" >&2; return 2; }
  case $2 in 0x19|0x1E|0x2D) ;; *) echo "hd_crd: refused register $2 (not a CPLD DIAG read)" >&2; return 2 ;; esac
  [ -n "$(hd_bus "$1")" ] || { echo "hd_crd: refused tray $1" >&2; return 2; }
  t=$(hd_left); [ "$t" -gt "${TRAY_HSC_READ_S:-3}" ] && t=${TRAY_HSC_READ_S:-3}; [ "$t" -ge 1 ] || return 0
  o=$(pg_run timeout -k 2 "$t" "${HD_IP[@]}" raw 0x06 0x52 "$(hd_bus "$1")" 0x80 0x01 "$2" < /dev/null 2> /dev/null) || return 0
  read -ra b <<< "$o"
  [ "${#b[@]}" = 1 ] && [[ ${b[0]} =~ ^[0-9a-fA-F]{2}$ ]] && echo "${b[0]}"
}
hd_diag() {  # <tray>: "cpld_diag ..." lines, the not-read debug fields, then one "cpld_diag summary: ..." line
  local f r nm g v i bad s="" n=0 dead=""
  for f in $HD_DIAG_RD; do
    IFS=: read -r r nm g <<< "$f"
    if [ -n "$dead" ]; then echo "cpld_diag reg $r $nm not read: no answer at reg $dead before it"; s="$s $nm not read;"; continue; fi
    if [ "$(hd_left)" -lt 1 ]; then echo "cpld_diag reg $r $nm not read: the ${TRAY_HSC_DUMP_S:-30} s dump budget is spent"; s="$s $nm not read;"; continue; fi
    v=$(hd_crd "$1" "$r")
    if [ -z "$v" ]; then echo "cpld_diag reg $r $nm not read: no answer to ${HD_IP[*]} raw 0x06 0x52 $(hd_bus "$1") 0x80 0x01 $r"; s="$s $nm not read;"; dead=$r; continue; fi
    bad=""; for i in 0 1 2 3 4 5 6 7; do [ $(( (16#$v >> i) & 1 )) = "$g" ] || bad="${bad:+$bad,}ASIC$((i + 1))"; done
    if [ -n "$bad" ]; then echo "cpld_diag reg $r $nm 0x$v (not good: $bad)"; else echo "cpld_diag reg $r $nm 0x$v (all 8 good)"; fi
    s="$s $nm 0x$v${bad:+ ($bad)};"
  done
  while read -r f; do [ -n "$f" ] && echo "CPLD DIAG dbg ${f%%:*} ${f#*:} not read: needs a CPLD write" && n=$((n + 1)); done < <(tr ' ' '\n' <<< "$HD_DIAG_DBG")
  echo "cpld_diag summary:${s%;}; $n debug-index fields not read (need a CPLD write)"
}
# hd_dump <file> <step> <"<tray> <missing BDFs...>" lines>: the dump of each listed tray not yet dumped in <step>, into
# <file>, synced; one say line per tray. Never fails (rc 0), so the glx_reset that follows always runs.
hd_dump() {
  local f=$1 st=$2 t bdfs cp todo="" l s trays=""
  while read -r t bdfs; do
    [[ $t =~ ^[1-4]$ ]] || continue
    [[ "|$HD_DONE|" == *"|$st:$t|"* ]] && continue
    HD_DONE="$HD_DONE|$st:$t"; todo="$todo$t ${bdfs:-}"$'\n'; trays="${trays:+$trays }$t"
  done <<< "$3"
  [ -n "$todo" ] || return 0
  HD_END=$(( $(date +%s) + ${TRAY_HSC_DUMP_S:-30} ))
  [ ${#HD_IP[@]} -gt 0 ] || hd_route || :
  local out
  out=$(
    echo "tray power state dump, read-only, before the glx_reset"
    echo "step: $st"
    echo "start: $(date '+%F %T %Z') (epoch $(date +%s)); budget ${TRAY_HSC_DUMP_S:-30} s"
    echo "route: ${HD_IP[*]:-none}${HD_WHY:+ ($HD_WHY)}"
    cp=""
    if [ ${#HD_IP[@]} = 0 ]; then cp="not read: $HD_WHY"
    elif [ ! -f "$HD_DIR/../cpld_5f.py" ]; then cp="not read: cpld_5f.py not found"
    else s=$(( $(hd_left) / 2 )); [ "$s" -gt 15 ] && s=15
      if [ "$s" -lt 1 ]; then cp="not read: the dump budget is spent"
      else cp=$(pg_run timeout -k 2 "$s" python3 -I "$HD_DIR/../cpld_5f.py" --ipmitool "${HD_IP[*]}" --trays "${trays// /,}" < /dev/null 2>&1) || cp="${cp}"$'\n'"not read: cpld_5f.py rc $? (timeout $s s)"; fi; fi
    while read -r t bdfs; do
      [ -n "$t" ] || continue
      echo "--- tray $t ($(hd_bdfs "$t"); missing: ${bdfs:-not on PCIe, BDFs not mapped}) bus $(hd_bus "$t")"
      l=$(grep -a "^tray $t bus " <<< "$cp" | head -1)
      if [ -n "$l" ]; then echo "cpld $l"
      else echo "cpld not read: $(grep -a '^not read: ' <<< "$cp" | tail -1 | sed 's/^not read: //' | grep . || echo "cpld_5f.py gave no tray $t line")"; fi
      if [ ${#HD_IP[@]} = 0 ]; then echo "hsc 0x40 0x42 not read: $HD_WHY"
      else hd_hsc "$t" 0x40; hd_hsc "$t" 0x42; fi
      if [ ${#HD_IP[@]} = 0 ]; then echo "cpld_diag not read: $HD_WHY"; echo "cpld_diag summary: not read"; else hd_diag "$t"; fi
    done <<< "$todo"
    echo "end: epoch $(date +%s)" 2>&1)
  printf '%s\n' "$out" >> "$f"
  sync "$f" 2> /dev/null || sync
  while read -r t bdfs; do
    [ -n "$t" ] || continue
    s=$(awk -v t="$t" '
      /^--- tray / { on = ($3 == t) } !on { next }
      /^hsc 0x4. / && $2 != last { if (last != "") h = h ";"; last = $2 }
      /^cpld tray / { c = $0; sub(/.*-> /, "", c); sub(/:.*/, "", c) }
      /^cpld not read/ { c = "not read" }
      /^hsc 0x4. STATUS_WORD 0x/ { h = h sprintf(" HSC %s STATUS_WORD %s", $2, $4) }
      /^hsc 0x4. READ_VIN 0x/ { h = h sprintf(" VIN %s V", $5) }
      /^hsc 0x4. READ_IOUT 0x/ { h = h sprintf(" IOUT %s A", $5) }
      /^hsc 0x4. STATUS_WORD not read/ { h = h sprintf(" HSC %s not read", $2) }
      /^hsc 0x40 0x42 not read/ { h = h " HSCs not read" }
      /^cpld_diag summary: / { d = $0; sub(/^cpld_diag summary: */, "", d) }
      END { printf "CPLD %s;%s; CPLD DIAG %s", c ? c : "not read", h, d ? d : "not read" }' <<< "$out" | sed 's/;;/;/g')
    say "reset: before the glx_reset ($st): tray $t ($(hd_bdfs "$t")) power state: $s (dump $f); the glx_reset runs anyway"
  done <<< "$todo"
  return 0
}
# hd_latch <file> <what>: before every reset (C2, C9, every recovery glx_reset) and every
# tray re-power (a power cycle of trays), STATUS_WORD (0x79) and STATUS_INPUT (0x7C) of all 8 tray HSCs, whatever chips are
# present: one line per HSC appended to <file> and synced, and one say line. "latched" names the set bits among
# VIN_UV_WARN, VIN_UV_FAULT (STATUS_INPUT bits 5, 4), OFF and IOUT_OC (STATUS_WORD bits 6, 4), or none; CML (bit 1) is
# named apart, never read into them. Reads only through hd_rd (no CLEAR_FAULTS: run.sh's one clear per run stays in
# lib/hsc_check.sh). Bounded by TRAY_HSC_LATCH_S (20 s) for all 16 reads; an HSC past it, silent, or with no route is
# "not read: <why>". Never fails (rc 0), so the reset or re-power that follows always runs.
hd_latch() {
  local f=$1 w=$2 t a b b0 b1 sw si k out cells bus
  HD_END=$(( $(date +%s) + ${TRAY_HSC_LATCH_S:-20} ))
  [ ${#HD_IP[@]} -gt 0 ] || hd_route || :
  out=$(
    echo "HSC latch read, read-only, before $w"
    echo "start: $(date '+%F %T %Z') (epoch $(date +%s)); budget ${TRAY_HSC_LATCH_S:-20} s; route: ${HD_IP[*]:-none}${HD_WHY:+ ($HD_WHY)}"
    for t in 1 2 3 4; do for a in 0x40 0x42; do
      printf 'tray %s (%s) hsc %s ' "$t" "$(hd_bdfs "$t")" "$a"
      if [ ${#HD_IP[@]} = 0 ]; then echo "not read: $HD_WHY"; continue; fi
      if [ "$(hd_left)" -lt 1 ]; then echo "not read: the ${TRAY_HSC_LATCH_S:-20} s latch budget is spent"; continue; fi
      bus=$(hd_bus "$t"); b=$(hd_rd "$t" "$a" 0x79 2)
      [ -n "$b" ] || { echo "not read: no answer to ${HD_IP[*]} raw 0x06 0x52 $bus $a 0x02 0x79"; continue; }
      read -r b0 b1 <<< "$b"; sw=$(( 16#$b0 | 16#$b1 << 8 )); si=""
      [ "$(hd_left)" -ge 1 ] && b=$(hd_rd "$t" "$a" 0x7C 1) && [ -n "$b" ] && si=$(( 16#$b ))
      k=""
      [ -n "$si" ] && [ $(( (si >> 5) & 1 )) = 1 ] && k="$k VIN_UV_WARN"
      [ -n "$si" ] && [ $(( (si >> 4) & 1 )) = 1 ] && k="$k VIN_UV_FAULT"
      [ $(( (sw >> 6) & 1 )) = 1 ] && k="$k OFF"
      [ $(( (sw >> 4) & 1 )) = 1 ] && k="$k IOUT_OC"
      b=$(hd_bits STATUS_WORD "$sw"); printf 'STATUS_WORD 0x%04x%s STATUS_INPUT ' "$sw" "${b:+ ($b)}"
      if [ -n "$si" ]; then b=$(hd_bits STATUS_INPUT "$si"); printf '0x%02x%s' "$si" "${b:+ ($b)}"; else printf 'not read'; fi
      echo " latched: ${k:- none}" | sed 's/:  */: /'
      [ $(( (sw >> 1) & 1 )) = 1 ] && echo "tray $t ($(hd_bdfs "$t")) hsc $a CML set (PMBus communication fault, not power)"
    done; done
    echo "end: epoch $(date +%s)")
  printf '%s\n' "$out" >> "$f"
  sync "$f" 2> /dev/null || sync
  cells=$(awk '$1 == "tray" && $4 == "hsc" {   # tray <t> (<bdfs>) hsc <addr> ...
      if ($6 == "CML") { cml = cml " t" $2 "/" $5; next }
      c = "t" $2 "/" $5
      if ($6 == "not") c = c " not read"
      else { si = ""; for (i = 8; i < NF; i++) if ($i == "STATUS_INPUT") si = $(i + 1)
        l = $0; sub(/.*latched: /, "", l); c = c " " $7 "/" (si == "not" ? "unread" : si) " " l }
      s = s (s == "" ? "" : ", ") c }
    END { printf "%s%s", s, cml != "" ? "; CML set:" cml : "" }' <<< "$out")
  say "reset: HSC latch before $w (STATUS_WORD/STATUS_INPUT): ${cells:-not read} (log $f); the reset runs anyway"
  return 0
}
