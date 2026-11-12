#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/link_check.sh: the host-link check of run.sh (every mode: tray-to-host PCIe faults).
# Sourced by run.sh only; defines functions, runs nothing by itself. Read-only sysfs, no hwmon, no device open:
#   lk_check pre|post   per Galaxy chip address (32: buses 01-08 tray 1, 41-48 tray 2, c1-c8 tray 3, 81-88 tray 4) a live
#                       read of the chip and its root port -> $OUT/link/snap_<label>.tsv, judged by fleet_py.py hostlink:
#                       off PCIe (not on the bus, config space ffff or unread), LnkSta unreadable (not trained), link
#                       slower or narrower than the design on the chip or root port side, or gone down since pre -> PCIE
#                       FAIL; above the design -> PCIE WARN. pre also saves the AER counters of every chip and its root
#                       port; post judges their rise over the run (fleet_py.py aerdelta: AER FAIL any uncorrectable, AER
#                       WARN correctable >= FLEET_AER_RATE_WARN/h and >= FLEET_AER_MIN events). Every line names the chip
#                       BDF, tray 1-4, root port and host. With --mode they are fleet findings (fl_find), else log
#                       lines; FAIL lines also go to fault.txt.
#                       pre holds the FAIL lines of a chip not answering at the start (held_pre.txt):
#                       lk_settle (lk_check start, after the start reset or the recount) decides, so a chip the start
#                       reset brings back is fl_c0settle's PREFLIGHT WARN, not a PCIE FAIL
#   lk_settle           after the start reset (rk_start) or the recount with none (fl_c0settle): lk_check start when pre held
#   lk_stop             before "Couldn't start" after a start reset: lk_check start (the links after it), then lk_hint;
#                       a --known-dropped-tray kd_settle did not name (KD_PEND=1) is judged with the rest
#   lk_hint             the fix per tray with a PCIE FAIL, first line bold: tray still powered (a chip of it answers, or
#                       its CPLD reads the 12 V good) -> the recovery ladder (lk_ladder: warm reset, glx_reset, glx_reset
#                       with retimer reset, BMC cold cycle, power drain, reseat, replace; steps this run tried marked,
#                       the next one named); a whole tray without power -> bus bar first
#   lk_report           after the RESULT line: without --mode the findings and rc 8 for a FAIL (when nothing worse);
#                       then lk_hint
# env: TRAY_REPRO_LINK_DESIGN (design per slot position, bus & 0x3f; default "auto": per slot the host's own majority link
#   at the start: some galaxies train every chip at 16 GT/s, quads 1-4 at 32 GT/s with x8 at 06/46/86/c6:00.0;
#   "32.0x1 6:32.0x8" pins the quads' design, from 352 C0 snapshots of quads 1-4, 2026-10-22),
#   FLEET_AER_RATE_WARN (6) and FLEET_AER_MIN (2): see README (AER thresholds)
LK_PY=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../fleet/fleet_py.py
LK_DONE=""; LK_LAST=""
lk_rdv() { local -n lk_rv=$1; lk_rv=$(timeout 2 cat "$2" 2> /dev/null) || { lk_rv=-; return 0; }; lk_rv=${lk_rv%% *}; lk_rv=${lk_rv:--}; }   # lk_rd into variable $1
lk_rd() { local v; lk_rdv v "$1"; echo "$v"; }
lk_snap() {  # one row per Galaxy chip address: bdf rp bus vendor speed width rp_speed rp_width ("-" = no file)
  local u k bus b d rp w sp wd br=""
  for d in "$SYS"/bus/pci/devices/*; do   # bridges by secondary bus: a chip off the bus still has its root port
    [ -e "$d/secondary_bus_number" ] && { w=; { read -r w < "$d/secondary_bus_number"; } 2> /dev/null; br="$br$w ${d##*/}"$'\n'; }; done
  for u in 0x00 0x40 0xc0 0x80; do for k in 1 2 3 4 5 6 7 8; do
    bus=$(( u + k )); b=0000:$(printf '%02x' $bus):00.0; d=$SYS/bus/pci/devices/$b
    # the known dropped tray (run.sh kd_check), named (KD_ON) or pending kd_settle (KD_PEND)
    { [ "${KD_ON:-0}" = 1 ] || [ "${KD_PEND:-0}" = 1 ]; } && [ "$(rk_tray "$b")" = "$KD_TRAY" ] && continue
    if [ -e "$d" ]; then
      rp=$(readlink -f "$d"); case $rp in */*) rp=${rp%/*}; rp=${rp##*/}; rp=${rp:-/} ;; *) rp=. ;; esac   # basename of dirname
      w=$(timeout 2 od -An -tx2 -N2 "$d/config" 2> /dev/null); w=${w//[[:space:]]/}
      lk_rdv sp "$d/current_link_speed"; lk_rdv wd "$d/current_link_width"
      printf '%s\t%s\ton\t%s\t%s\t%s' "$b" "$rp" "${w:-unread}" "$sp" "$wd"
    else
      rp=$(awk -v n=$bus '$1 == n { print $2; exit }' <<< "$br")
      printf '%s\t%s\toff\t-\t-\t-' "$b" "${rp:--}"
    fi
    if [ -n "$rp" ] && [ -e "$SYS/bus/pci/devices/$rp" ]; then
      lk_rdv sp "$SYS/bus/pci/devices/$rp/current_link_speed"; lk_rdv wd "$SYS/bus/pci/devices/$rp/current_link_width"
      printf '\t%s\t%s\n' "$sp" "$wd"
    else printf '\t-\t-\n'; fi
  done; done
}
lk_find() {  # one finding (once): fleet findings with --mode, else the log; FAIL lines also to fault.txt
  grep -qxF -- "$*" "$OUT/link/findings.txt" 2> /dev/null && return 0
  echo "$*" >> "$OUT/link/findings.txt"
  if [ -n "${FL_OUT:-}" ]; then fl_find "$*"; else say "$*"; fi
  case $* in *" FAIL "*) echo "$(date '+%F %T %Z') $*" >> "$OUT/fault.txt" ;; esac
}
lk_held() {  # $1 line, $2 pre snapshot: a FAIL line of a chip not answering config space at the start (held for lk_settle)
  case $1 in "PCIE FAIL "*) ;; *) return 1 ;; esac
  awk -F'\t' -v b="0000:$(echo "$1" | cut -d' ' -f3)" '$1 == b && !($3 == "on" && $4 == "1e52") { f = 1 } END { exit !f }' "$2"
}
lk_check() {  # $1 pre|post|start (pre once: run.sh without --mode, fl_c0 with it; AER only pre and post)
  local d=$OUT/link l a=(); mkdir -p "$d"; case " $LK_DONE " in *" $1 "*) return 0 ;; esac; LK_DONE="$LK_DONE $1"
  lk_snap > "$d/snap_$1.tsv"; cut -d' ' -f1 /proc/uptime > "$d/uptime_$1"; LK_LAST=$d/snap_$1.tsv
  [ "$1" != pre ] && [ -s "$d/snap_pre.tsv" ] && a=("$d/snap_pre.tsv")
  python3 "$LK_PY" hostlink "$d/snap_$1.tsv" "${TRAY_REPRO_LINK_DESIGN:-auto}" "$H" "${a[@]}" 2>> "$d/judge.err" |
    while read -r l; do
      if [ "$1" = pre ] && lk_held "$l" "$d/snap_pre.tsv"; then echo "$l" >> "$d/held_pre.txt"; else lk_find "$l"; fi; done
  [ "$1" = pre ] && [ -s "$d/held_pre.txt" ] && say "host link pre: $(cut -d' ' -f3-5 "$d/held_pre.txt" | sort -u | paste -sd ' ')" \
    "not answering at the start; their link lines wait for the start reset (lk_check start decides; $d/held_pre.txt)"
  [ "$1" = start ] || python3 "$LK_PY" aer "$SYS" $(awk -F'\t' '$3 == "on" && $4 == "1e52" { print $1 }' "$d/snap_$1.tsv") > "$d/aer_$1.tsv" 2>> "$d/judge.err"
  if [ "$1" = post ] && [ -s "$d/aer_pre.tsv" ]; then
    python3 "$LK_PY" aerdelta "$d/aer_pre.tsv" "$d/aer_post.tsv" "${FLEET_AER_RATE_WARN:-6}" \
      "$(awk -v a="$(cat "$d/uptime_pre")" -v b="$(cat "$d/uptime_post")" 'BEGIN { printf "%.4f", (b - a) / 3600 }')" \
      "${FLEET_AER_MIN:-2}" "$H" 2>> "$d/judge.err" | while read -r l; do
        # a chip the per-stage delta already reported (fleet): INFO, not a second finding
        if [ -s "${FL_OUT:-/nonexistent}/aer_stage/reported" ] && grep -qxF "${l%%, host *}" "$FL_OUT/aer_stage/reported"; then
          say "AER INFO whole run, already reported per stage: $l"; else lk_find "$l"; fi; done
  fi
  say "host link $1: $(awk -F'\t' '$3 == "on" && $4 == "1e52"' "$d/snap_$1.tsv" | grep -c .)/${RK_WANT:-32} chips answer config space," \
      "$(cat "$d/findings.txt" 2> /dev/null | grep -c '^PCIE FAIL') PCIe link FAIL line(s) so far; $d"
  [ -s "$d/judge.err" ] && say "host link $1: the judge failed, links not judged: $(tail -1 "$d/judge.err")"
  return 0
}
lk_ladder() {  # $1 tray: the recovery ladder for a chip off the bus or a link fault on a powered tray, the steps this run
  # already tried (RK_RESETS, run.sh's reset record) marked, the first one not tried named next. Prints only:
  # run.sh itself runs step 1 at most.
  local t=$1 m1="" next
  # tt-smi -r (warm reset) is never a step: glx_reset is step 1; "NO RESET (...)" in RK_RESETS is not a try
  case ${RK_RESETS:-} in *"glx_reset (tt-smi -glx_reset) FAILED"*|*"tray re-power (tt-smi -glx_reset) FAILED"*) m1=" [tried this run: failed]" ;;
    *"glx_reset (tt-smi -glx_reset)"*|*"then tray re-power"*) m1=" [tried this run]" ;; esac
  if [ -z "$m1" ]; then next="1. glx_reset"; else next="2. glx_reset with retimer reset"; fi
  echo "FIX tray $t: **next: $next** (steps 1-3 remote, 4-6 on site)"
  echo "  1. glx_reset$m1 -> 2. glx_reset with retimer reset -> 3. single-host BMC cold cycle"
  echo "  on site: 4. complete power drain (on site, or a networked PDU if the site has one) -> 5. reseat tray $t -> 6. replace tray $t if it drops again"
}
lk_hint() {  # the fix per tray with a PCIE FAIL (once); fault.txt too
  local d=$OUT/link s=$LK_LAST t on why cp st fix
  [ -s "$d/findings.txt" ] && [ ! -e "$d/fix.txt" ] || return 0
  for t in $(sed -n 's/^PCIE FAIL [^ ]* (tray \([1-4]\)).*/\1/p' "$d/findings.txt" | sort -u); do
    [ "${KD_ON:-0}" = 1 ] && [ "$t" = "$KD_TRAY" ] && continue   # named the known drop by kd_settle: no fix line
    on=$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import fleet_py as f
n = sum(1 for l in open(sys.argv[2]) for r in [l.rstrip("\n").split("\t")]
        if len(r) == 8 and r[2] == "on" and r[3] == "1e52" and f.ubb(r[0]) == "tray " + sys.argv[3])
print(n)' "$(dirname "$LK_PY")" "$s" "$t" 2> /dev/null)
    cp=$OUT/cpld_reg1e_dbg5f.txt
    [ "${on:-0}" = 0 ] && [ ! -s "$cp" ] && declare -F cpld_drop > /dev/null && cpld_drop "$OUT" 1   # whole tray off: its 12 V, live
    st=$(grep -a -m1 '^SUMMARY ' "$cp" 2> /dev/null | grep -oE "tray ?$t(/UBB[0-3])? [^ ;]+ [A-Z0-9_]+" | awk '{ print $NF }')
    if [ "${on:-0}" -gt 0 ]; then why="tray $t powered: ${on} of 8 chips answer"; fix=reseat
    else case $st in
      HEALTHY|CHIPS_DOWN_12V_GOOD) why="tray $t powered: no chip answers, CPLD $st (12 V good)"; fix=reseat ;;
      ALIVE) why="tray $t powered: no chip answers, CPLD reg 0x1E ff (all chips alive)"; fix=reseat ;;
      BUSBAR_SIG|TRAY_12V_LOST|HOTSWAP_OC|TRAY_OFF) why="tray $t without power: CPLD $st"; fix=busbar ;;
      *) why="tray $t: no chip answers, CPLD ${st:-not read}: a whole-tray loss"; fix=busbar ;;
    esac; fi
    { if [ $fix = reseat ]; then lk_ladder "$t"
      else echo "FIX tray $t: **1. Replace the bus bar -> 2. if the tray still drops, replace tray $t**"; fi
      echo "  why: host $H, $why; $(grep -a "^PCIE FAIL [^ ]* (tray $t)" "$d/findings.txt" | sed 's/^PCIE FAIL //; s/, host [^:]*//' | sort -u | head -8 | paste -sd ';' | sed 's/;/; /g')"
    } >> "$d/fix.txt"
  done
  touch "$d/fix.txt"; [ -s "$d/fix.txt" ] || return 0
  cat "$d/fix.txt"; sed "s/^/$(date '+%F %T %Z') /" "$d/fix.txt" >> "$OUT/fault.txt"
}
lk_settle() { [ -s "$OUT/link/held_pre.txt" ] || return 0; lk_check start; }
lk_stop() {  # a tray still pending kd_settle is not the known drop: its chips are judged now
  local kp=${KD_PEND:-0} b f=${FL_OUT:-/nonexistent}/findings.txt
  LK_DONE=${LK_DONE// start/}   # judge again: a chip C2 dropped after a held-chip recovery gets its PCIE FAIL (lk_find: once)
  local KD_PEND=0; lk_check start
  # KD_PEND: fl_c0settle named tray N's missing chips 'chip K: not answering' while lk_check skipped them; the PCIE FAIL
  # line now names each of them, so that C0 line alone goes (every other PREFLIGHT FAIL stays)
  [ "$kp" = 1 ] && [ -s "$f" ] && for b in $(sed -n "s/^PCIE FAIL \([^ ]*\) (tray $KD_TRAY).*/\1/p" "$OUT/link/findings.txt" 2> /dev/null | sort -u); do
    sed -i "/^PREFLIGHT FAIL ${b//./\\.} (tray $KD_TRAY) chip [0-9?]*: not answering\$/d" "$f"; done   # '?': no tt-kmd node
  lk_hint
}
lk_report() {  # after the RESULT line (and fl_report): the findings without --mode, rc 8 for a FAIL; then the fix
  local f=$OUT/link/findings.txt
  if [ -z "${FLEET:-}" ] && [ -s "$f" ]; then
    echo "host link: $(grep -c . "$f") finding(s); $OUT/link"; cat "$f"
    [ "$RK_RC" = 0 ] && grep -qv ' WARN ' "$f" && RK_RC=8
  fi
  lk_hint; return 0
}
