#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/bmc_heat.sh: heat and power evidence around a bus-bar or tray drop (every drop so far
# was found without it). Sourced by run.sh after lib/heavy_logs.sh (needs heavy_logs_init's HL_SUDO,
# HL_IPMI_OK and HL_SDRC); functions only.
#   bh_poll_start <run dir> / bh_poll_stop [quiet]   background in-band SDR poll every BH_POLL_S (2 s, or as fast as the
#       BMC answers) to <run dir>/bmc_heat_poll.csv: every sensor of type Temperature or Fan (VR, HSC, DIMM, CPU, NIC,
#       PSU, brick, ASIC, retimer, inlet/outlet; no longer a fixed list), plus by name PSU, tray, HSC, fan and
#       total W and P12V (BH_SDR_RE). The Temperature/Fan names come from one `sdr type Temperature` and `sdr type Fan`
#       at the start; a reading in degrees C or RPM is kept also when those failed. Each read's rc, ms and rows go to
#       bmc_heat_poll_reads.tsv, so the summary says when the poll could not run and why (off, timed out, failed). Rows "utc,epoch,sensor,tray,value,unit,status": sensor is the
#       BMC's raw name (evidence), tray is "tray N" (UBBk = tray k+1), empty for box-wide sensors. run.sh starts it with
#       --heavy-logs, in every mode (bh_poll_want; no longer --mode=full by itself, full = plain, longer). Stops with the run: run.sh's
#       end, fault_exit on any exit, and by itself once run.sh is gone; killing it also kills its current read.
#   bh_drop <capture dir> [label]   at each drop capture, in every mode: one SDR snapshot (bmc_sdr_<label>.txt raw,
#       bmc_heat_<label>.csv mapped) and each tray CPLD's alive (reg 0x1E) and VR PG (reg 0x19) to cpld_vrpg.txt; says
#       one line for each, by tray 1-4.
# BMC use is reads only. Every ipmitool call of the poll and snapshot goes through bh_guard, which allows only
# `[-S cache] sdr elist|list|get|type` and `sensor list|get|reading` and refuses anything else (raw, chassis, power,
# thresh, set, ...). The CPLD read reuses lib/heavy_logs.sh's hl_reg (master write-read of one register, the same
# read-only path as cpld_5f.py and the light poll); no CPLD debug-index write, no power action. Never agera2-app;
# never any power limit (TDP) write.
# env: BH_POLL_S (2), BH_SDR_RE (sensors kept by name beside every Temperature and Fan one), HEAVY_IPMITOOL (ipmitool binary, as lib/heavy_logs.sh)
BH_PID=""; BH_CSV=""; BH_OFF=""; BH_TYPES=""
BH_SDR_RE=${BH_SDR_RE:-'^(TEMP_INLET|TEMP_OUTLET|TEMP_PSU[0-9]+|Power_PSU[0-9]+|Power_UBB[0-3]|Power_MB_HSC|Power_HSC|Power_FAN|Power_Total|TMAX_UBB[0-3]_(ASIC|RT)|TEMP_UBB[0-3]_RT[0-9]+|Fan[A-Za-z0-9_]*RPM|FAN[A-Za-z0-9_]*|P12V[A-Za-z0-9_]*)$'}
BH_BUS=(0x11 0x13 0x0D 0x0F)   # tray 1-4 CPLD bus (at 0x80), as cpld_5f.py TRAY_BUS

bh_say() { if declare -F say > /dev/null; then say "$*"; else echo "$*"; fi; }
bh_guard() {  # <ipmitool args>: 0 for a sensor read form only, else 2 (logged to stderr); nothing else reaches ipmitool
  local a=("$@") x
  [ "${a[0]:-}" = -S ] && [ -n "${a[1]:-}" ] && a=("${a[@]:2}")
  case "${a[0]:-} ${a[1]:-}" in
    "sdr elist"|"sdr list"|"sdr get"|"sdr type"|"sensor list"|"sensor get"|"sensor reading") ;;
    *) echo "bmc_heat: refused ipmitool $* (reads only: sdr/sensor)" >&2; return 2 ;;
  esac
  for x in "${a[@]:2}"; do
    case ${x,,} in raw|chassis|power|thresh|set|reset|cycle|off|on|-s|-f|exec|shell|dump|fill|lan|user|mc|bmc)
      echo "bmc_heat: refused ipmitool $* (word '$x')" >&2; return 2 ;; esac
  done
  return 0
}
bh_ipmi() {  # <ipmitool args>: one guarded read, 20 s cap (exec: the caller's pid is the read itself, so a kill ends it)
  bh_guard "$@" || return 2
  exec timeout -k 2 20 ${HL_SUDO:+$HL_SUDO} "${HEAVY_IPMITOOL:-ipmitool}" "$@"
}
bh_ok() { [ "${HL_IPMI_OK:-0}" = 1 ]; }   # in-band ipmitool usable (heavy_logs_init: sudo -n or root, and /dev/ipmi*)
bh_why() { echo "${HL_IPMI_WHY:-no in-band ipmitool: needs sudo -n or root, ipmitool and /dev/ipmi*}"; }   # why not bh_ok
bh_names() { awk -F'|' 'NF >= 5 { n = $1; gsub(/^ +| +$/, "", n); if (n != "") print n }'; }   # sdr list on stdin -> names
bh_rows() {  # <utc> <epoch>: sdr elist on stdin -> mapped CSV rows of the sensors kept (no header): BH_SDR_RE by name,
  # every name bh_types listed (Temperature, Fan), and any reading in degrees C or RPM
  awk -F'|' -v t="$1" -v e="$2" -v re="$BH_SDR_RE" -v ty="${BH_TYPES:-/dev/null}" '
    BEGIN { while ((getline k < ty) > 0) keep[k] = 1 }
    { n = $1; gsub(/^ +| +$/, "", n); st = $3; gsub(/^ +| +$/, "", st); r = $5; gsub(/^ +| +$/, "", r)
      if (n == "" || (n !~ re && !(n in keep) && r !~ /^-?[0-9.]+ (degrees C|RPM)$/)) next
      tr = ""; if (match(n, /UBB[0-3]/)) tr = "tray " (substr(n, RSTART + 3, 1) + 1)
      v = ""; u = ""; if (r ~ /^-?[0-9.]+( |$)/) { v = r; sub(/ .*/, "", v); u = r; sub(/^[^ ]+ ?/, "", u) }
      printf "%s,%s,%s,%s,%s,%s,%s\n", t, e, n, tr, v, u, st }'
}
bh_line() {  # <mapped csv rows on stdin>: one line by tray 1-4: box inlet/outlet/P12V/total, then each tray's W, ASIC, RT
  awk -F, '
    function put(k, x) { if (x != "" && (!(k in m) || x + 0 > m[k] + 0)) m[k] = x }
    function low(k, x) { if (x != "" && (!(k in m) || x + 0 < m[k] + 0)) m[k] = x }
    $3 == "TEMP_INLET" { put("in", $5) } $3 == "TEMP_OUTLET" { put("out", $5) } $3 ~ /^P12V(_ADC)?$/ { low("p12", $5) }
    $3 == "Power_Total" { put("tot", $5) } $3 ~ /^TEMP_PSU/ { put("psut", $5) } $3 ~ /^Power_PSU/ { psu[$3] = $5 }
    $4 != "" && $3 ~ /^Power_UBB/ { put($4 " W", $5) } $4 != "" && $3 ~ /_ASIC$/ { put($4 " ASIC", $5) }
    $4 != "" && $3 ~ /_RT$/ { put($4 " RT", $5) }
    $6 == "degrees C" && $5 != "" { nt++; if (ht == "" || $5 + 0 > hv + 0) { ht = $3; hv = $5 } }
    END { s = "inlet " (("in" in m) ? m["in"] " C" : "-") ", outlet " (("out" in m) ? m["out"] " C" : "-")
          s = s ", P12V " (("p12" in m) ? m["p12"] " V" : "-") ", total " (("tot" in m) ? m["tot"] " W" : "-")
          s = s ", PSU max " (("psut" in m) ? m["psut"] " C" : "-")
          for (t = 1; t <= 4; t++) { k = "tray " t
            s = s "; " k " " ((k " W" in m) ? m[k " W"] " W" : "- W") " ASIC " ((k " ASIC" in m) ? m[k " ASIC"] " C" : "-") \
                " RT " ((k " RT" in m) ? m[k " RT"] " C" : "-") }
          if (ht != "") s = s "; hottest sensor " ht " " hv " C"
          print s }'
}
bh_sdr_args() { BH_A=(${HL_SDRC:+-S "$HL_SDRC"} sdr elist); }
bh_loop() {  # <csv> <main pid> [types file]: the poll; TERM/INT/HUP kill the read or sleep in flight, then exit. First
  # the names of every Temperature and Fan sensor into the types file (one `sdr type` each), which bh_rows then keeps
  local f=$1 main=$2 ty=${3:-} c="" tmp t e s0 dt rc nr
  tmp=$(mktemp "${TMPDIR:-/tmp}/bh_sdr.XXXXXX") || return 1
  trap '[ -n "$c" ] && kill "$c" 2>/dev/null; rm -f "$tmp" "$tmp.err"; exit 0' TERM INT HUP
  trap 'rm -f "$tmp" "$tmp.err"' EXIT
  bh_sdr_args
  if [ -n "$ty" ]; then : > "$ty"
    for t in Temperature Fan; do
      ( bh_ipmi ${HL_SDRC:+-S "$HL_SDRC"} sdr type "$t" ) > "$tmp" 2>/dev/null & c=$!; wait "$c"; c=""; bh_names < "$tmp" >> "$ty"
    done; BH_TYPES=$ty; fi
  while kill -0 "$main" 2>/dev/null; do
    s0=$(date +%s%N); t=$(date -u +%FT%T.%3NZ); e=$(date +%s.%3N)
    ( bh_ipmi "${BH_A[@]}" ) > "$tmp" 2> "$tmp.err" & c=$!; wait "$c"; rc=$?; c=""
    nr=$(bh_rows "$t" "$e" < "$tmp" | tee -a "$f" | wc -l)
    printf '%s\t%s\t%s\t%s\n' "$e" "$rc" $(( ($(date +%s%N) - s0) / 1000000 )) "$nr" >> "${f%.csv}_reads.tsv"
    [ "$rc" != 0 ] && [ -s "$tmp.err" ] && head -1 "$tmp.err" > "${f%.csv}_err.txt"
    dt=$(( ${BH_POLL_S:-2} * 1000 - ($(date +%s%N) - s0) / 1000000 ))   # every BH_POLL_S, or at once if the read was slower
    [ "$dt" -gt 0 ] && { sleep "$(printf '%d.%03d' $((dt / 1000)) $((dt % 1000)))" & c=$!; wait "$c"; c=""; }
  done
}
bh_poll_want() { [ "${HEAVY:-0}" = 1 ]; }   # --heavy-logs only, in every mode (full = plain)
bh_poll_start() {  # <run dir>: the background poll (run.sh calls it only when bh_poll_want)
  [ -z "$BH_PID" ] || return 0
  if ! bh_ok; then BH_OFF=$(bh_why); bh_say "BMC heat poll: off ($BH_OFF)"; return 0; fi
  declare -F hl_sdr_cache > /dev/null && [ -n "${HL_OUT:-}" ] && hl_sdr_cache   # sdr elist ~1 s instead of 7-14 s
  BH_CSV=$1/bmc_heat_poll.csv; echo "utc,epoch,sensor,tray,value,unit,status" > "$BH_CSV"
  printf 'epoch\trc\tms\trows\n' > "${BH_CSV%.csv}_reads.tsv"; BH_TYPES=$1/bmc_heat_types.txt   # bh_loop fills it
  ( trap - EXIT; bh_loop "$BH_CSV" "$$" "$BH_TYPES" ) > /dev/null 2>&1 & BH_PID=$!
  bh_say "BMC heat poll: every ${BH_POLL_S:-2} s (or as fast as the BMC answers) to $BH_CSV"
}
bh_poll_stop() {  # [quiet]: stops the poll this lib started, by its pid; says its summary unless quiet: the samples and
  # max line, or that it could not run and why (off: no ipmitool, no root or sudo, no /dev/ipmi*; reads timed out or
  # failed), and the read interval: a poll every ~2 s cannot see a 50 ms burst
  if [ -z "$BH_PID" ]; then [ -z "${1:-}" ] && [ -n "$BH_OFF" ] && bh_say "BMC heat poll: could not run ($BH_OFF): no BMC temperature, fan or power samples in this run"; return 0; fi
  kill "$BH_PID" 2>/dev/null; wait "$BH_PID" 2>/dev/null; BH_PID=""
  [ -n "${1:-}" ] || [ ! -s "$BH_CSV" ] && return 0
  local n r k=${BH_CSV%.csv}; n=$(awk -F, 'NR > 1 { s[$2] = 1 } END { print length(s) }' "$BH_CSV")
  r=$(awk -F'\t' 'NR > 1 { a++; if (a == 1) e0 = $1; e1 = $1; if ($2 == 124 || $2 == 137) to++; else if ($4 == 0 && $2 != 0) bad++; else if ($4 == 0) emp++ }
    END { if (!a) { print "no read finished"; exit }
          w = a " reads" (a > 1 ? sprintf(", one every %.1f s", (e1 - e0) / (a - 1)) : "")
          if (to) w = w ", " to " timed out (20 s cap)"; if (bad) w = w ", " bad " failed"; if (emp) w = w ", " emp " with no sensor kept"
          print w }' "${k}_reads.tsv" 2> /dev/null)
  if [ "$n" = 0 ]; then
    bh_say "BMC heat poll: could not run (${r:-no read finished}$([ -s "${k}_err.txt" ] && echo ": $(cat "${k}_err.txt")")): no BMC temperature, fan or power samples in this run; $BH_CSV"
    return 0; fi
  bh_say "BMC heat poll: $n samples ($r; blind to bursts shorter than that, e.g. 50 ms), max over the run: $(awk 'NR > 1' "$BH_CSV" | bh_line); $BH_CSV"
}
bh_snapshot() {  # <dir> <label>: one guarded sdr elist, raw and mapped; says the line by tray
  local d=$1 l=$2 t e
  t=$(date -u +%FT%T.%3NZ); e=$(date +%s.%3N); bh_sdr_args
  ( bh_ipmi "${BH_A[@]}" ) > "$d/bmc_sdr_$l.txt" 2>&1
  { echo "utc,epoch,sensor,tray,value,unit,status"; bh_rows "$t" "$e" < "$d/bmc_sdr_$l.txt"; } > "$d/bmc_heat_$l.csv"
  bh_say "BMC at $l: $(awk 'NR > 1' "$d/bmc_heat_$l.csv" | bh_line) ($d/bmc_heat_$l.csv)"
}
bh_vrpg() {  # <dir>: each tray CPLD's alive (0x1E: ff alive, 00 dead) and VR PG (0x19: ff all rails good), by tray 1-4
  local d=$1 k a p n s="" t; t=$(date -u +%FT%T.%3NZ)
  for k in 1 2 3 4; do
    a=$(hl_reg "${BH_BUS[$((k - 1))]}" 0x80 0x1E); p=$(hl_reg "${BH_BUS[$((k - 1))]}" 0x80 0x19); a=${a,,}; p=${p,,}
    case $a in ff) n=alive ;; 00) n="CPLD DEAD (12 V lost?)" ;; "") n="no answer" ;; *) n="alive?" ;; esac
    case $p in ff) n="$n, rails good" ;; "") n="$n, vr_pg no answer" ;; *) n="$n, RAILS DOWN" ;; esac
    echo "$t tray $k bus ${BH_BUS[$((k - 1))]} reg0x1E=${a:--} reg0x19=${p:--} $n" >> "$d/cpld_vrpg.txt"
    s="$s${s:+; }tray $k alive ${a:--} vr_pg ${p:--} ($n)"
  done
  bh_say "CPLD at drop: $s ($d/cpld_vrpg.txt)"
}
bh_drop() {  # <capture dir> [label]: the drop's BMC snapshot and CPLD alive/vr_pg; never changes the run's result
  local d=$1 l=${2:-drop}; mkdir -p "$d"
  if ! bh_ok; then echo "no in-band ipmitool ($(bh_why))" > "$d/bmc_heat_skipped.txt"
    bh_say "BMC/CPLD at drop: skipped (no in-band ipmitool: $(bh_why))"; return 0; fi
  bh_snapshot "$d" "$l"; declare -F hl_reg > /dev/null && bh_vrpg "$d"; return 0
}
