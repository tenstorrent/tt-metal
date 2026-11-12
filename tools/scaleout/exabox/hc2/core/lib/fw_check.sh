#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/fw_check.sh: C0 firmware uniformity. Reads each chip's firmware versions from sysfs only, with no
# device open: tt_fw_bundle_ver, and every other tt_*_ver tt-kmd has (cm/eth/m3 firmware where present). A chip whose
# version differs from the other chips' is named before anything opens a device: UMD refuses the mesh open on such a host
# ("Firmware bundle version mismatch for device <asic id>": one chip on 19.12.0.0, the other 31 chips on
# 19.16.0.0, which run.sh reported then as "chip not pinpointed").
# Sourced by run.sh (uses SYS, rk_tray, say, OUT); also runs alone, read-only:
#   bash lib/fw_check.sh [sysfs root]   the table, then one FAIL line per odd chip; exit 0 uniform, 1 a mismatch,
#                                       2 no chip or no version read
#   fw_table              "<bdf> <kmd> <asic_id> <attr>=<value>..." per Blackhole function ("-" where unread)
#   fw_uniform <table>    per attr, each chip off the most common value:
#                         "PREFLIGHT FAIL 47:00.0 (tray 2) chip 14: tt_fw_bundle_ver 19.12.0.0, 31 of 32 chips 19.16.0.0"
#   fw_preflight          run.sh: $OUT/fw.txt, the FAIL lines to the log and fault.txt (FW_FAIL keeps them for fleet's
#                         findings), else one summary line
declare -F rk_tray > /dev/null || rk_tray() {  # standalone: run.sh's own (bus & 0xF0 -> tray 1-4; 0000:88:00.0 or 88:00.0)
  local a=${1#"${1%%[![:space:]]*}"}; a=${a%%[[:space:]]*}; a=${a##*/}   # first word; a sysfs path keeps its last part; no fork
  if [[ $a =~ ^([0-9a-fA-F]{4,}:)?([0-9a-fA-F]{2}):[0-9a-fA-F]{2}\.[0-7]$ ]]; then
    case $(( 16#${BASH_REMATCH[2]} & 0xF0 )) in
      0) echo 1 ;; 64) echo 2 ;; 192) echo 3 ;; 128) echo 4 ;; *) echo "?" ;; esac
  else echo "ERROR: PCI address '$1' has no bus byte (want 0000:88:00.0 or 88:00.0): tray 1-4 unknown, named tray ?" >&2; echo "?"; fi
}
fw_table() {
  local d n f v o
  for d in "${SYS:-/sys}"/bus/pci/devices/*; do
    v=; { read -r v < "$d/vendor"; } 2>/dev/null; [ "$v" = 0x1e52 ] || continue   # builtins where no read can hang
    for n in "$d"/tenstorrent/tenstorrent!*; do [ -e "$n" ] && break; n=; done
    v=${n##*!}; [[ $v =~ ^[0-9]+$ ]] || v=-; o="${d##*/} $v"
    v=; [ -n "$n" ] && v=$(timeout 2 cat "$n/tt_asic_id" 2>/dev/null); v=${v//[$' \n']/}; o="$o ${v:--}"
    for f in ${n:+"$n"/tt_*_ver}; do
      [ -e "$f" ] || continue; v=$(timeout 2 cat "$f" 2>/dev/null); v=${v//[$' \n']/}; [ -n "$v" ] && o="$o ${f##*/}=$v"
    done
    echo "$o"
  done
}
fw_uniform() {
  local b c a v r nr n
  awk '{ for (i = 4; i <= NF; i++) { split($i, kv, "="); k = kv[1]; v = substr($i, length(k) + 2)
           n[k]++; c[k, v]++; if (!((k, v) in seen)) { seen[k, v] = 1; vs[k] = vs[k] " " v } row[NR, k] = v }
         b[NR] = $1; m[NR] = $2; has[NR] = ($2 != "-"); tot += ($2 != "-") }
       END { for (k in n) {
               split(substr(vs[k], 2), l, " "); best = ""
               for (j in l) if (best == "" || c[k, l[j]] > c[k, best] || (c[k, l[j]] == c[k, best] && l[j] > best)) best = l[j]
               for (i = 1; i <= NR; i++) {
                 if ((i, k) in row) { if (row[i, k] != best) print b[i], m[i], k, row[i, k], best, c[k, best], n[k] }
                 else if (has[i]) print b[i], m[i], k, "-", best, c[k, best], tot } } }' "$1" |
    sort -k3,3 -k1,1 |
    while read -r b c a v r nr n; do
      if [ "$v" = - ]; then   # a chip with a node but no readable version while others have one (old firmware, read timeout)
        echo "PREFLIGHT FAIL ${b#0000:} (tray $(rk_tray "$b")) chip $c: $a unread, $nr of $n chips $r"; continue; fi
      echo "PREFLIGHT FAIL ${b#0000:} (tray $(rk_tray "$b")) chip $c: $a $v, $nr of $n chips $r$([ $(( nr * 2 )) -gt "$n" ] || echo " (no majority)")"
    done
}
# fw_notready <table>: "<bdf> <chip> <bundle>" per chip whose firmware did not come up: tt_fw_bundle_ver
# 0.0.0.0 (seen on one tray: UMD then waits 300 s at every open, "Firmware startup error ...
# postcode=0xc0de0014"), or no bundle version on a chip with a tt-kmd node while the other chips have one
fw_notready() {
  awk '{ v[NR] = "-"; for (i = 4; i <= NF; i++) if ($i ~ /^tt_fw_bundle_ver=/) { v[NR] = substr($i, 18); any = 1 }
         b[NR] = $1; m[NR] = $2 }
       END { for (i = 1; i <= NR; i++) if (v[i] == "0.0.0.0" || (v[i] == "-" && m[i] != "-" && any)) print b[i], m[i], v[i] }' "$1"
}
fw_major() {  # <table>: the most common tt_fw_bundle_ver other than 0.0.0.0 ("-" when none)
  awk '{ for (i = 4; i <= NF; i++) if ($i ~ /^tt_fw_bundle_ver=/ && $i != "tt_fw_bundle_ver=0.0.0.0") c[substr($i, 18)]++ }
       END { for (k in c) if (c[k] > n) { n = c[k]; b = k }; print (b == "" ? "-" : b) }' "$1"
}
FW_LIB=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")" 2> /dev/null && pwd)
# fw_postcode <bdf> [sysfs root]: the ARC postcode of a chip whose firmware is not ready, for its FW NOT READY line. sysfs
# has none and a device open waits 300 s for the firmware, so lib/bar_read.py reads it (and BOOT_STATUS, ERROR_STATUS0)
# from a read-only mmap of resource0, bounded by FW_PC_TIMEOUT (4 s, killed 1 s later: under 5 s per chip).
# FW_PC_CMD overrides the reader (tests). Healthy chips read postcode 0xc0de0041, boot status 0x5 (32/32 chips on one box).
fw_postcode() {
  local t=${FW_PC_TIMEOUT:-4} o rc pc bs es x
  local -a c=(${FW_PC_CMD[@]+"${FW_PC_CMD[@]}"})
  if [ "${#c[@]}" = 0 ]; then c=(python3 -I "$FW_LIB/bar_read.py"); [ "$(id -u)" = 0 ] || c=(sudo -n "${c[@]}"); fi
  o=$(timeout -k 1 "$t" "${c[@]}" "$@" 2>&1); rc=$?
  case $rc in
    0) ;;
    124|137) echo "postcode read timed out (${t} s)"; return ;;
    *) echo "postcode not read (BAR read rc $rc: $(head -1 <<< "$o" | cut -c1-120))"; return ;;
  esac
  pc=$(grep -o 'postcode=[^ ]*' <<< "$o" | cut -d= -f2); bs=$(grep -o 'boot_status=[^ ]*' <<< "$o" | cut -d= -f2)
  es=$(grep -o 'error_status0=[^ ]*' <<< "$o" | cut -d= -f2); x=$(grep -o 'pcie_x=[^ ]*' <<< "$o" | cut -d= -f2)
  case $pc in
    0xffffffff|0xFFFFFFFF) echo "postcode unreadable (0xffffffff)" ;;
    0x*) echo "postcode $pc (boot status $bs, error status0 $es)" ;;
    -) echo "postcode not read (PCIe tile at NOC x $x: ARC not on BAR0)" ;;
    *) echo "postcode not read (BAR read: $(head -1 <<< "$o" | cut -c1-120))" ;;
  esac
}
# fw_nr_settle <when>: once, after run.sh's goal-9 start reset (fleet modes: lib/fleet.sh's fl_c2/fl_start). Reads the
# versions again; a chip whose firmware is ready now gets its PREFLIGHT line turned into a recovered WARN, a chip still
# not ready prints 'FAIL tray N' at once with one line per chip under it (BDF, chip, fw bundle, fw_postcode) and stays in
# FW_NR, which keeps it out of every device open (fl_c3's TT_VISIBLE_DEVICES) and skips the stages that need the 4x8 mesh
fw_nr_settle() {
  [ -n "${FW_NR:-}" ] && [ -z "${FW_NR_DONE:-}" ] || return 0
  FW_NR_DONE=1
  local t=$OUT/fw_settle.txt now b c v nv u k maj f=${FL_OUT:-$OUT}/findings.txt l still=""
  fw_table > "$t"; now=$(fw_notready "$t"); maj=$(fw_major "$t")
  while read -r b c v; do
    [ -n "$b" ] || continue
    if grep -q "^$b " <<< "$now"; then still+="$b $c $(grep "^$b " <<< "$now" | awk '{ print $3 }')"$'\n'; continue; fi
    nv=$(awk -v b="$b" '$1 == b { for (i = 4; i <= NF; i++) if ($i ~ /^tt_fw_bundle_ver=/) print substr($i, 18) }' "$t")
    l="PREFLIGHT WARN ${b#0000:} (tray $(rk_tray "$b")) chip $c: recovered: firmware not ready at the start (tt_fw_bundle_ver $v), ready $1 (tt_fw_bundle_ver ${nv:--})"
    [ -f "$f" ] && grep -v "^PREFLIGHT FAIL ${b#0000:} " "$f" > "$f.nr" && mv "$f.nr" "$f"
    FW_FAIL=$(grep -v "^PREFLIGHT FAIL ${b#0000:} " <<< "${FW_FAIL:-}")
    if declare -F fl_find > /dev/null; then fl_find "$l"; else say "$l"; fi
  done <<< "$FW_NR"
  FW_NR=$(grep . <<< "$still")
  # the settled table's uniformity again: a chip that recovered to a minority version keeps (or gets) its PREFLIGHT FAIL
  # the old lines leave findings.txt and fault.txt, the lines for the settled table go in
  local old=${FW_FAIL:-} nl ft=${OUT}/fault.txt
  nl=$(fw_uniform "$t")
  if [ "$nl" != "$old" ]; then
    while read -r l; do [ -n "$l" ] || continue
      for k in "$f" "$ft"; do [ -f "$k" ] && { grep -vF -- "$l" "$k" > "$k.nr"; mv "$k.nr" "$k"; }; done
    done <<< "$old"
    while read -r l; do [ -n "$l" ] || continue
      if declare -F fl_find > /dev/null; then fl_find "$l"; else say "$l"; fi
      echo "$(date '+%F %T %Z') $l" >> "$ft"
    done <<< "$nl"
  fi
  FW_FAIL=$nl
  [ -n "$FW_NR" ] || { say "C0 firmware: every chip not ready at the start is ready $1"; return 0; }
  for u in 1 2 3 4; do
    k=$(while read -r b c v; do [ "$(rk_tray "$b")" = "$u" ] && echo "$b"; done <<< "$FW_NR" | grep -c .)
    [ "$k" -gt 0 ] || continue
    say "FAIL tray $u ($k chip$([ "$k" -gt 1 ] && echo s) firmware not ready $1; left out of every device stage)"
    while read -r b c v; do
      [ "$(rk_tray "$b")" = "$u" ] || continue
      l="FW NOT READY ${b#0000:} (tray $u) chip $c: firmware not ready $1: tt_fw_bundle_ver $v (the other chips $maj), $(fw_postcode "$b"); left out of every device stage"
      say "    $l"; echo "$(date '+%F %T %Z') $l" >> "$OUT/fault.txt"
      # the finding under FAIL tray N at the end: the PREFLIGHT FAIL line already there (fl_c0), else this one
      grep -q "^PREFLIGHT FAIL ${b#0000:} " "$f" 2> /dev/null || { if declare -F fl_find > /dev/null; then fl_find "PREFLIGHT FAIL ${b#0000:} (tray $u) chip $c: $l"; fi; }
    done <<< "$FW_NR"
  done
}
fw_preflight() {
  local l b c v
  fw_table > "$OUT/fw.txt"; FW_FAIL=$(fw_uniform "$OUT/fw.txt"); FW_NR=$(fw_notready "$OUT/fw.txt"); FW_NR_DONE=""
  if [ -n "${KD_TRAY:-}" ]; then   # --known-dropped-tray=N: a tray N chip that reads ffff (every version 255.255.255.255
    # or unread) is the known drop, not a firmware mismatch; a live tray N chip is still compared
    awk -v kd="$KD_TRAY" '{ t = tolower(substr($1, 6, 1)); t = (t == "0" ? 1 : t == "4" ? 2 : t == "c" ? 3 : t == "8" ? 4 : 0)   # as rk_tray
      dead = 1; for (i = 4; i <= NF; i++) if ($i !~ /=255\.255\.255\.255$/) dead = 0
      if (!(t == kd && dead)) print }' "$OUT/fw.txt" > "$OUT/fw_kd.txt"
    FW_FAIL=$(fw_uniform "$OUT/fw_kd.txt"); FW_NR=$(fw_notready "$OUT/fw_kd.txt")
  fi
  [ -n "$FW_NR" ] && say "C0 firmware not ready on $(grep -c . <<< "$FW_NR") chip(s) ($(while read -r b c v; do echo "${b#0000:} (tray $(rk_tray "$b")) $v"; done <<< "$FW_NR" | paste -sd, | sed 's/,/, /g')): no device open reaches them; the start reset first, then FAIL tray N for each still not ready"
  if [ -n "$FW_FAIL" ]; then
    while read -r l; do say "$l"; echo "$(date '+%F %T %Z') $l" >> "$OUT/fault.txt"; done <<< "$FW_FAIL"
  elif ! grep -q '=' "$OUT/fw.txt"; then say "C0 firmware: no tt_*_ver in sysfs ($(grep -c . "$OUT/fw.txt") chips): not compared"
  else say "C0 firmware: uniform on $(grep -c '=' "$OUT/fw.txt") chips ($(awk '{ for (i = 4; i <= NF; i++) print $i }' "$OUT/fw.txt" | sort -u | paste -sd' '))"; fi
}
if [ "${BASH_SOURCE[0]:-$0}" = "$0" ]; then   # run alone (also piped: bash -s < lib/fw_check.sh)
  SYS=${1:-${SYS:-/sys}}; t=$(fw_table); echo "$t"
  grep -q '=' <<< "$t" || { echo "no chip or no tt_*_ver read under $SYS" >&2; exit 2; }
  f=$(fw_uniform <(echo "$t")); [ -z "$f" ] && { echo "PASS firmware uniform on $(grep -c '=' <<< "$t") chips"; exit 0; }
  echo "$f"; exit 1
fi
