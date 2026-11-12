#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

declare -F pg_track > /dev/null || . "$(dirname "${BASH_SOURCE[0]}")/pgroups.sh"   # pg_run (device calls under timeout)
declare -F sudo_l > /dev/null || . "$(dirname "${BASH_SOURCE[0]}")/sudo_l.sh"   # exact sudo -n -l probes
# lib/heavy_logs.sh: run.sh's evidence logging (the old run-thorough.sh's logging, folded into run.sh). Every run
# takes light snapshots and the raw power poll; the slow BMC reads (marked [full] below) run with
# HEAVY_FULL=1 (run.sh --heavy-logs) and in every snapshot once a fault is named (FC=1, lib/fault_capture.sh). Sourced by run.sh; every function is time-bounded, logs each step to <outdir>/heavy/steps.tsv
# and returns 0, so it never fails or changes the test run. Read-only: nothing here resets, re-powers, signals or
# writes to a chip or a power path. No BMC, CPLD or PMBus write: the CPLD debug registers need an index
# write first, so they are named 'not read: needs a CPLD index write (not allowed)'. The only write outside <outdir> is,
# only with HEAVY_HWMON_GUARD=1, the mode of a hung
# chip's hwmon files (hl_hwmon_guard; saved first, undo script written). BMC: in-band ipmitool reads (KCS, sudo -n) and Redfish
# GETs only; never agera2-app. BMC credentials are read from $HEAVY_BMC_ENV (BMC_USER, BMC_PASSWORD) by reference and
# reach curl on stdin, never argv, a log or a file. Reads that open a chip (tt-smi -s, ETH counters, exalens) run only
# when no process holds /dev/tenstorrent.
#   heavy_logs_init <outdir>            once: versions, firmware (Redfish inventory, CPLD versions), static PCIe tree
#   heavy_logs_snapshot <phase>         live state: hung chips (lspci rev ff, 'Timeout waiting for ARC response' in the
#                                       kernel log; never hwmon-read), hwmon file modes, AER + link per chip and root
#                                       port, lspci -vvv, dmesg/journal delta, MCE, holders; with the device free: tt-smi
#                                       -s, ETH link counters. [full] sysfs telemetry x2 (heartbeat), SEL + delta, SDR,
#                                       CPLD dump + 0x5f, Redfish power/thermal (about 40 s; light about 6 s). HEAVY_HWMON_GUARD=1 (run.sh --hwmon-guard; non-serving hosts only):
#                                       chmod 000 a hung chip's hwmon *_input/*_max (hl_hwmon_guard), the one write to sysfs
#   heavy_logs_on_hang <bdf...>         a hang or loss, before any reset: noc_snapshot.sh (lspci -vvv RP+EP, tt-triage
#                                       NOC0 then NOC1), dump_fast_dispatch, exalens probe NOC0/1, then a snapshot
#   heavy_logs_bmc_poll_start / _stop   every HEAVY_POLL_S (2 s) to a CSV: UBB0-3 and HSC power and each tray CPLD's
#                                       vr_pg (raw ipmitool, about 0.4 s a round); [full] UBB power, P12V, PSU and temps
#                                       from the SDR (sdr elist, about 9 s a round, keeps the BMC busy)
#   heavy_logs_finish                   one-page index.md: each artefact with ok or FLAG
# env: HEAVY_SYS (default $TRAY_REPRO_SYS or /sys), HEAVY_IPMITOOL (ipmitool), HEAVY_BMC_ENV
# (~/.config/quad-access/bmc.env), HEAVY_BMC_HOST (default: from ipmitool lan print 1), HEAVY_POLL_S (2),
# HEAVY_PY (python with tt_umd; default python3), HEAVY_AS (prefix to run chip reads as the non-root user),
# HEAVY_TRIAGE (tt-triage command; noc_snapshot.sh's search otherwise), HEAVY_TRIAGE_TIMEOUT_S (120),
# HEAVY_WANT (chips expected, 32), HEAVY_FULL (1; run.sh sets 0 without --heavy-logs), HEAVY_HWMON_GUARD (0), HEAVY_RUN_USER (the run's user; default $SUDO_USER or id -un), HEAVY_DEVICE_READS (auto: only when /dev/tenstorrent is free; 0 never), HEAVY_REDFISH (1; 0 off).
HL_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")"); HL_ROOT=$(dirname "$HL_DIR")
HL_OUT=""; HL_HUNG=""; HL_N=0; HL_POLL_PID=""; HL_SUDO=""; HL_LSUDO=""; HL_XSUDO=""; HL_IPMI_OK=0; HL_IPMI_WHY=""; HL_BMC=""; HL_RF=0; HL_SDRC=""
HL_SDR_RE='Power_UBB[0-3] |PSU|P12V|54V|TEMP_UBB[0-3]_RT0|TMAX_UBB'
HL_BUSES="0x0D 0x0F 0x11 0x13"   # tray CPLDs at 0x80 (buses 0x0D 0x0F 0x11 0x13 = trays 3 4 1 2, cpld_5f.py TRAY_BUS); PDB CPLD bus 0x09 at 0x62
hl_sys() { echo "${HEAVY_SYS:-${TRAY_REPRO_SYS:-/sys}}"; }
hl_have() { command -v "$1" > /dev/null 2>&1; }
hl_ts() { date '+%F %T %Z'; }
hl_step() {  # $1 phase, $2 step, $3 output file, $4 timeout s, rest: the command (stdout+stderr to the file)
  # HL_DEADLINE (epoch s; run.sh sets it while its run budget is on): the step's timeout is cut to end by then,
  # and past it the step is skipped (rc "skipped" in steps.tsv); each skip, and each cut step that then timed out (its
  # output lost), is listed in budget_cut.tsv. A cut step that ends inside its cut lost nothing: budget_capped.tsv only
  # (one quick pass logged 7 such 0-1 s steps as cut and gave a healthy 32/32 run INCOMPLETE)
  local ph=$1 st=$2 f=$3 to=$4 t0 rc l cut=""; shift 4; t0=$(date +%s)
  if [ -n "${HL_DEADLINE:-}" ]; then l=$(( HL_DEADLINE - t0 ))
    if [ "$l" -lt 1 ]; then echo "skipped: past the run budget's line ($HL_DEADLINE) that keeps the C9 reset step's time free" > "$f"
      printf '%s\t%s\tskipped\t0\t%s\n' "$ph" "$st" "${f#"$HL_OUT"/}" >> "$HL_OUT/steps.tsv"; printf '%s\t%s\tskipped\n' "$ph" "$st" >> "$HL_OUT/budget_cut.tsv"; return 0; fi
    [ "$l" -lt "$to" ] && { cut="cut to $l s of $to s"; to=$l; }
  fi
  pg_run timeout -k 5 "$to" "$@" > "$f" 2>&1 < /dev/null; rc=$?   # pg_run: a TERM reaps it (tt-smi, triage, exalens)
  if [ -n "$cut" ]; then case $rc in 124|137) printf '%s\t%s\t%s: timed out\n' "$ph" "$st" "$cut" >> "$HL_OUT/budget_cut.tsv" ;;
    *) printf '%s\t%s\t%s: ended in %s s\n' "$ph" "$st" "$cut" "$(( $(date +%s) - t0 ))" >> "$HL_OUT/budget_capped.tsv" ;; esac; fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$ph" "$st" "$rc" "$(( $(date +%s) - t0 ))" "${f#"$HL_OUT"/}" >> "$HL_OUT/steps.tsv"
  return 0
}
hl_fn() {  # like hl_step for a function of this lib: a fresh bash under timeout gets the hl_* functions and settings
  local ph=$1 st=$2 f=$3 to=$4 q; shift 4; q=$(printf '%q ' "$@")
  # shellcheck disable=SC2046
  hl_step "$ph" "$st" "$f" "$to" bash -c "$(declare -f $(compgen -A function hl_)); $(declare -p $(compgen -v HL_; compgen -v HEAVY_; compgen -v TRAY_REPRO_SYS) 2>/dev/null); $q"
}
hl_ib() { ${HL_SUDO:+$HL_SUDO} "${HEAVY_IPMITOOL:-ipmitool}" "$@"; }   # in-band ipmitool (KCS)
hl_reg() { timeout 15 ${HL_SUDO:+$HL_SUDO} "${HEAVY_IPMITOOL:-ipmitool}" raw 0x06 0x52 "$1" "$2" 0x01 "$3" 2>/dev/null | tr -d ' \n'; }
hl_scan() {  # "<comm> <pid> <user> <fd> CHR <node>" per open /dev/tenstorrent fd; /proc only, so a dead NFS mount cannot hang it (lsof and fuser stat every mount)
  local d=${HEAVY_DEVD:-/dev/tenstorrent} pr=${HEAVY_PROC:-/proc} l n p c u
  ${HL_XSUDO:+$HL_XSUDO} timeout 20 find "$pr"/[0-9]*/fd -maxdepth 1 -lname "$d/*" -printf '%p %l\n' 2>/dev/null | while read -r l n; do
    p=${l#"$pr"/}; p=${p%%/*}; c=$(cat "$pr/$p/comm" 2>/dev/null) || continue
    u=$(stat -c %U "$pr/$p" 2>/dev/null); l=${l##*/}
    echo "${c:-?} $p ${u:-?} ${l}u CHR $n"
  done
}
hl_holders() {  # processes holding a /dev/tenstorrent node (empty: free)
  hl_scan
}
hl_dev_free() { [ "${HEAVY_DEVICE_READS:-auto}" != 0 ] && [ -z "$(hl_holders)" ]; }
hl_chips() {  # "<bdf> <root port bdf> <chip>" per Blackhole function, from sysfs
  local s d b rp c v p g; s=$(hl_sys)
  for d in "$s"/bus/pci/devices/*; do
    v=""; { read -r v < "$d/vendor"; } 2>/dev/null
    [ "$v" = 0x1e52 ] || continue
    b=${d##*/}; p=$(readlink -f "$d"); p=${p%/*}; rp=${p##*/}; c=""
    for g in "$d"/tenstorrent/*!*; do [ -e "$g" ] && c=${g##*!}; break; done
    echo "$b $rp ${c:-?}"
  done
}
hl_bus() { [[ $1 =~ ^([0-9a-fA-F]{4,}:)?([0-9a-fA-F]{2}):[0-9a-fA-F]{2}\.[0-7]$ ]] && echo "${BASH_REMATCH[2]}"; }  # bus byte of 0000:88:00.0 or 88:00.0, empty if none
hl_tray() { local b; b=$(hl_bus "$1"); [ -n "$b" ] || { echo "ERROR: PCI address '$1' has no bus byte: tray 1-4 unknown, named tray ?" >&2; echo "?"; return; }; local n=$(( 16#$b & 0xF0 )); case $n in 0) echo 1;; 64) echo 2;; 192) echo 3;; 128) echo 4;; *) echo "?";; esac; }

# ------------------------------------------------------------------ captures (each writes stdout)
hl_versions() {
  local s; s=$(hl_sys)
  hl_ts; date -u '+%FT%TZ'; uname -a; grep PRETTY /etc/os-release; echo "cmdline: $(cat /proc/cmdline)"
  echo "kmd: $(cat "$s/module/tenstorrent/version" 2>/dev/null)"
  for t in tt-smi tt-flash; do hl_have $t && echo "$t: $(timeout 20 $t --version 2>&1 | tail -1)"; done
  echo "tt-metal: ${TT_METAL_HOME:-unset} $(git -C "${TT_METAL_HOME:-/nonexistent}" rev-parse --short HEAD 2>/dev/null)"
  echo "tray-test: $(git -C "$HL_ROOT" rev-parse --short HEAD 2>/dev/null) $(git -C "$HL_ROOT" status --porcelain 2>/dev/null | wc -l) changed"
  for f in bios_version bios_date board_name product_name product_serial; do
    echo "dmi $f: $(cat "$s/class/dmi/id/$f" 2>/dev/null || ${HL_XSUDO:+$HL_XSUDO} cat "$s/class/dmi/id/$f" 2>/dev/null)"; done
  echo "hugepages: $(grep -E 'HugePages_(Total|Free)|Hugepagesize' /proc/meminfo | tr -s ' ' | tr '\n' ' ')"
  for d in "$s"/class/tenstorrent/tenstorrent!*; do   # per chip firmware as the KMD reports it
    [ -e "$d" ] || continue
    echo "${d##*/} $(basename "$(readlink -f "$d/device")") $(for f in "$d"/tt_*fw*ver* "$d"/tt_serial "$d"/tt_card_type; do [ -r "$f" ] && printf '%s=%s ' "${f##*/}" "$(cat "$f" 2>/dev/null)"; done)"
  done
  hl_have ras-mc-ctl && echo "rasdaemon: present" || echo "rasdaemon: not installed (MCE from dmesg/journal only)"
  return 0
}
hl_cpld_ver() {  # CPLD firmware version bytes (reg 0x00-0x03) per tray and the PDB
  local b; for b in $HL_BUSES; do echo "tray-bus $b: $(for r in 0x00 0x01 0x02 0x03; do printf '%s ' "$(hl_reg "$b" 0x80 $r)"; done)"; done
  echo "PDB: $(for r in 0x00 0x01 0x02 0x03; do printf '%s ' "$(hl_reg 0x09 0x62 $r)"; done)"
  return 0
}
hl_cpld_dump() {  # run_repro.sh's cpld_dump, reads only: alive, therm_trip, VR PG per tray, PDB plain registers
  local b r line
  echo "# $(date -u +%FT%T.%3NZ) tray reg0x1E alive (ff ok, 00 dead), reg0x2D therm_trip, reg0x19 VR PG; debug registers (tray dbg0x58 0x5F 0x59 0xD0 0xD1 0xD6 0x5D 0xD5 0xD2 0xD3, PDB dbg0xD1 0xD2 0xD4 0xD5 0x58 0x5F) not read: needs a CPLD index write (not allowed)"
  for b in $HL_BUSES; do
    line="tray-bus $b:"
    for r in 0x1E 0x2D 0x19; do line+=" reg$r=$(hl_reg "$b" 0x80 $r)"; done
    echo "$(date -u +%T.%3N) $line"
  done
  line="PDB:"; for r in 0x05 0x06 0x07 0x08 0x09 0x0A; do line+=" reg$r=$(hl_reg 0x09 0x62 $r)"; done
  echo "$(date -u +%T.%3N) $line"
  return 0
}
hl_telem() {  # per chip sysfs tt_* and hwmon, read twice 1 s apart: a heartbeat that does not move is flagged.
              # No hwmon reads of a chip in $HL_HUNG (hl_hung)
  local s d f b pass hw; s=$(hl_sys)
  for pass in 1 2; do
    [ $pass = 2 ] && sleep 1
    for d in "$s"/class/tenstorrent/tenstorrent!*; do
      [ -e "$d" ] || continue
      echo "== pass $pass ${d##*/} $(basename "$(readlink -f "$d/device")")"
      b=$(basename "$(readlink -f "$d/device")")
      if [[ " ${HL_HUNG:-} ${MG_HUNG:-} " == *" $b "* ]]; then   # MG_HUNG: the MCE guard's hung or held chips   # a hung chip: each hwmon read is an ARC message that times out
        echo "hwmon not read: $b is hung (hung_chips.txt)"; hw=()
      else hw=("$d"/device/hwmon/hwmon*/*_input "$d"/device/hwmon/hwmon*/*_label); fi
      for f in "$d"/tt_* "${hw[@]}"; do
        [ -r "$f" ] && echo "${f#"$d"/} $(timeout 2 cat "$f" 2>/dev/null)"; done
    done
  done
  return 0
}
hl_rd() { local -n _o=$1; _o=""; { read -r _o < "$2"; } 2>/dev/null || true; }   # one file's first line into a variable, no fork
hl_aer() {  # per chip and its root port: link speed/width (and max) and every nonzero AER counter
  local b rp c f a s n x cs cw ms mw k line last w; s=$(hl_sys)
  local -a ws
  while read -r b rp c; do
    [ -n "$b" ] || continue
    n=$(hl_bus "$b"); [ -n "$n" ] || { echo "ERROR: PCI address '$b' has no bus byte: tray 1-4 unknown, named tray ?" >&2; n=-1; }
    [ "$n" = -1 ] || n=$(( 16#$n & 0xF0 ))
    case $n in 0) n=1;; 64) n=2;; 192) n=3;; 128) n=4;; *) n="?";; esac
    for f in "$s/bus/pci/devices/$b" "$s/bus/pci/devices/$rp"; do
      if [ "${f##*/}" = "$b" ]; then x="chip $c tray $n"; else x="root port of $b"; fi
      echo "== ${f##*/} $x"
      hl_rd cs "$f/current_link_speed"; hl_rd cw "$f/current_link_width"; hl_rd ms "$f/max_link_speed"; hl_rd mw "$f/max_link_width"
      echo "link $cs x$cw (max $ms x$mw)"
      for a in aer_dev_correctable aer_dev_nonfatal aer_dev_fatal aer_rootport_total_err_cor aer_rootport_total_err_fatal aer_rootport_total_err_nonfatal; do
        [ -r "$f/$a" ] || continue
        while IFS= read -r line || [ -n "$line" ]; do
          read -ra ws <<< "$a $line"; last=${ws[${#ws[@]}-1]}
          [ "$last" != 0 ] && echo "$a $line"
        done < "$f/$a"
      done
    done
  done < <(hl_chips)
  return 0
}
hl_lspci_rp() { local rp; for rp in $(hl_chips | awk '{print $2}' | sort -u); do timeout 10 ${HL_LSUDO:+$HL_LSUDO} lspci -vvv -s "$rp"; done; return 0; }
hl_hwmon_files() {  # "<bdf> <file>": each chip's hwmon *_input and *_max, as /sys/class/hwmon/hwmonN/... where that exists
  local s b h n f; s=$(hl_sys)
  while read -r b _; do
    for h in "$s/bus/pci/devices/$b"/hwmon/hwmon*; do
      [ -d "$h" ] || continue; n=${h##*/}; [ -d "$s/class/hwmon/$n" ] && h=$s/class/hwmon/$n
      for f in "$h"/*_input "$h"/*_max; do [ -e "$f" ] && echo "$b $f"; done
    done
  done < <(hl_chips)
}
hl_hwmon_modes() {  # "<bdf> <mode> <file>" per hwmon *_input/*_max: 0 (000) = the hung-chip guard is on for that file
  local b f; while read -r b f; do echo "$b $(stat -c %a "$f" 2>/dev/null || echo '?') $f"; done < <(hl_hwmon_files); return 0
}
hl_hung() {  # <snapshot dir>: "<bdf> <why>" per hung chip: rev ff in lspci_vvv_chips.txt (config space reads all ones), or
             # 'Timeout waiting for ARC response' in journal_kernel.txt (the kernel log since the last snapshot)
  { sed -nE 's/^(([0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]) .*\(rev ff\).*/\1 lspci-rev-ff/p' "$1/lspci_vvv_chips.txt" 2>/dev/null
    grep -aoE 'tenstorrent [0-9a-f]{4}:[0-9a-f:.]+: Timeout waiting for ARC response' "$1/journal_kernel.txt" 2>/dev/null |
      awk '{ sub(/:$/, "", $2); print $2, "dmesg-ARC-timeout" }'
  } | awk '{ b = ($1 ~ /^[0-9a-f]{4}:/) ? $1 : "0000:" $1; if (!seen[b, $2]++) w[b] = (b in w) ? w[b] "," $2 : $2 }
           END { for (b in w) print b, w[b] }' | sort
}
hl_holder_users() {  # the user of each process holding a /dev/tenstorrent node, one per line
  hl_scan | awk '{ print $3 }'
}
hl_alias() {  # the ttp machines alias of this host's quad
  case $(hostname -s | tr '[:upper:]' '[:lower:]') in
    *) echo "<alias>" ;;
  esac
}
hl_hwmon_guard() {  # <snapshot dir>: only with HEAVY_HWMON_GUARD=1 (run.sh --hwmon-guard). chmod 000 the hwmon *_input and
  # *_max of each chip in hung_chips.txt, so non-root pollers (node_exporter, sensors) stop sending it ARC messages; root
  # readers are not stopped (DAC override). Non-serving hosts only: refused while a process of another user (an H3
  # server) holds /dev/tenstorrent. Each file's mode is saved first (hwmon_guard.saved); hwmon_guard_undo.sh restores it.
  # Prints the ledger line to record (notes/hold-on-hang.md). Never resets, signals or writes to a chip.
  [ "${HEAVY_HWMON_GUARD:-0}" = 1 ] && [ -s "$1/hung_chips.txt" ] || return 0
  local g=$HL_OUT/hwmon_guard.txt sv=$HL_OUT/hwmon_guard.saved me others b f cb m n=0 bad=0 done_b=""
  me=${HEAVY_RUN_USER:-${SUDO_USER:-$(id -un)}}
  others=$(hl_holder_users | grep -vxF "$me" | sort -u | tr '\n' ' ')
  if [ -n "${others// /}" ]; then
    echo "$(hl_ts) refused: /dev/tenstorrent is held by user(s) $others(not this run's user $me): a serving host? no chmod" | tee -a "$g"
    return 0
  fi
  while read -r b _; do
    while read -r cb f; do
      [ "$cb" = "$b" ] || continue
      m=$(stat -c %a "$f" 2>/dev/null) || continue
      [ "$m" = 0 ] && continue   # already guarded (an earlier snapshot): its saved mode stays the original
      grep -qF " $f" "$sv" 2>/dev/null || echo "$m $f" >> "$sv"
      if timeout 10 ${HL_XSUDO:+$HL_XSUDO} chmod 000 "$f" 2>> "$g"; then n=$((n + 1)); else bad=$((bad + 1)); fi
    done < <(hl_hwmon_files)
    done_b="$done_b $b"
  done < "$1/hung_chips.txt"
  [ -s "$sv" ] && awk '{ print "[ -e " $2 " ] && chmod " $1 " " $2 }' "$sv" > "$HL_OUT/hwmon_guard_undo.sh"
  echo "$(hl_ts) hung:$done_b; chmod 000 on $n hwmon file(s)$([ $bad = 0 ] || echo ", $bad failed"); undo: sudo bash $HL_OUT/hwmon_guard_undo.sh" >> "$g"
  [ $n = 0 ] && return 0
  local h p=""; h=$(hostname -s | tr '[:upper:]' '[:lower:]')
  for b in $done_b; do p="$p${p:+ && }lspci -s $b | grep -q . && ! lspci -s $b | grep -q rev.ff"; done   # back and not rev ff
  { echo "hwmon guard: chmod 000 on $n hwmon *_input/*_max file(s) of hung chip(s)$done_b. Record it with ttp:"
    echo "  ttp machines change add $(hl_alias) \"hwmon guard on $h:$done_b (hung; run.sh --hwmon-guard, $HL_OUT)\"" \
      "--undo \"bin/quad-ssh $h 'sudo bash $HL_OUT/hwmon_guard_undo.sh'\" --until-probe \"bin/quad-ssh $h '$p'\""
  } | tee -a "$g"
  return 0
}
hl_trays() {
  echo "# chips on PCIe per tray (bus & 0xF0: 00 tray 1, 40 tray 2, c0 tray 3, 80 tray 4)"
  timeout 30 lspci -D -d 1e52: | awk '{print $1}' | while read -r b; do echo "tray $(hl_tray "$b")"; done | sort | uniq -c
  echo "total $(timeout 30 lspci -d 1e52: | wc -l)"
  return 0
}
hl_eth() {  # ETH link counters of every chip's ETH cores (x 20-33, y 25), one JSON line per core; opens the chips
  local c specs=""; for c in $(hl_chips | awk '$3 != "?" {print $3}' | sort -n); do
    for x in $(seq 20 33); do specs+=" $c:$x,25"; done; done
  [ -n "$specs" ] || { echo "no chips"; return 0; }
  # shellcheck disable=SC2086
  ${HEAVY_AS:-} "${HEAVY_PY:-python3}" "$HL_DIR/ethctr.py" $specs
}
hl_redfish() {  # $1 Redfish path: one GET, credentials on stdin (curl -K -), never in argv
  ( . "${HEAVY_BMC_ENV:-$HOME/.config/quad-access/bmc.env}" 2>/dev/null || exit 3
    printf 'user = "%s:%s"\n' "${BMC_USER:-}" "${BMC_PASSWORD:-}" ) |
    curl -sk --max-time 20 -K - "https://$HL_BMC$1"
}

# ------------------------------------------------------------------ the interface
heavy_logs_init() {  # <outdir>
  HL_OUT=${1:?heavy_logs_init needs an out dir}/heavy; mkdir -p "$HL_OUT/static" || { HL_OUT=""; return 0; }
  printf 'phase\tstep\trc\tseconds\tfile\n' > "$HL_OUT/steps.tsv"; HL_N=0; hl_sdr_cache_drop
  # exact sudo -n -l probes, never sudo -n true. HL_SUDO: ipmitool (every form sent is a fixed line of
  # the fixed sudoers list under docs/, probed by the first one); HL_LSUDO: lspci; HL_XSUDO: the rest (find, chmod, cat, journalctl,
  # dmesg, ras-mc-ctl take varying arguments, so only root or NOPASSWD: ALL)
  HL_SUDO=$(sudo_pfx "${HEAVY_IPMITOOL:-ipmitool}" mc info); HL_LSUDO=$(sudo_pfx lspci -vvv -d 1e52:); HL_XSUDO=""
  if [ "$(id -u)" != 0 ]; then sudo_all && HL_XSUDO="sudo -n"; [ -n "$HL_SUDO" ] || HL_SUDO=none; fi
  [ "$HL_SUDO" = none ] && HL_SUDO="" && HL_IPMI_OK=0 || HL_IPMI_OK=1
  hl_have "${HEAVY_IPMITOOL:-ipmitool}" && ls "${HEAVY_IPMID:-/dev/ipmi}"* > /dev/null 2>&1 || HL_IPMI_OK=0
  HL_IPMI_WHY=""; hl_have "${HEAVY_IPMITOOL:-ipmitool}" || HL_IPMI_WHY="no ipmitool"   # why, for the summary
  ls "${HEAVY_IPMID:-/dev/ipmi}"* > /dev/null 2>&1 || HL_IPMI_WHY="${HL_IPMI_WHY:+$HL_IPMI_WHY, }no ${HEAVY_IPMID:-/dev/ipmi}*"
  [ "$(id -u)" != 0 ] && [ -z "$HL_SUDO" ] && HL_IPMI_WHY="${HL_IPMI_WHY:+$HL_IPMI_WHY, }no root and no passwordless sudo (sudo -n)"
  local o=$HL_OUT/static
  hl_fn init versions "$o/versions.txt" 60 hl_versions
  hl_step init lspci_tree "$o/lspci_tree.txt" 30 lspci -tv
  hl_step init lspci_vvv_chips "$o/lspci_vvv_chips.txt" 60 ${HL_LSUDO:+$HL_LSUDO} lspci -vvv -d 1e52:
  if [ $HL_IPMI_OK = 1 ]; then
    hl_step init ipmi_mc_info "$o/ipmi_mc_info.txt" 30 ${HL_SUDO:+$HL_SUDO} "${HEAVY_IPMITOOL:-ipmitool}" mc info
    hl_step init ipmi_fru "$o/ipmi_fru.txt" 60 ${HL_SUDO:+$HL_SUDO} "${HEAVY_IPMITOOL:-ipmitool}" fru
    hl_step init ipmi_lan "$o/ipmi_lan.txt" 30 ${HL_SUDO:+$HL_SUDO} "${HEAVY_IPMITOOL:-ipmitool}" lan print 1
    hl_fn init cpld_versions "$o/cpld_versions.txt" 120 hl_cpld_ver
  else
    echo "in-band ipmitool not usable (${HL_IPMI_WHY:-no ipmitool, or no root / passwordless sudo}): no SEL, SDR, FRU or CPLD" > "$o/ipmi_skipped.txt"
  fi
  HL_BMC=${HEAVY_BMC_HOST:-$(awk -F': *' '/^IP Address  /{print $2; exit}' "$o/ipmi_lan.txt" 2>/dev/null)}
  HL_RF=0
  if [ "${HEAVY_REDFISH:-1}" = 1 ] && [ -n "$HL_BMC" ] && [ -r "${HEAVY_BMC_ENV:-$HOME/.config/quad-access/bmc.env}" ] && hl_have curl; then
    hl_fn init redfish_tasks "$o/redfish_tasks.json" 25 hl_redfish /redfish/v1/TaskService/Tasks
    if grep -q '"Members"' "$o/redfish_tasks.json" 2>/dev/null; then
      if grep -q '"Members@odata.count": *0' "$o/redfish_tasks.json"; then HL_RF=1
      else echo "a BMC task is listed (an update may be running): no Redfish reads this run" > "$o/redfish_skipped.txt"; fi
    else echo "Redfish not reachable at the BMC (VPN?)" > "$o/redfish_skipped.txt"; fi
  else echo "Redfish skipped: no BMC address, no readable creds file, or HEAVY_REDFISH=0" > "$o/redfish_skipped.txt"; fi
  if [ $HL_RF = 1 ]; then
    local p n
    for p in Managers/bmc Systems/system UpdateService/FirmwareInventory/bios_active UpdateService/DevFirmware; do
      n=$(echo "$p" | tr '/' '_'); hl_fn init "redfish_$n" "$o/redfish_$n.json" 30 hl_redfish "/redfish/v1/$p"
    done
    for p in $(grep -o '"@odata.id": *"/redfish/v1/UpdateService/DevFirmware/[^"]*"' "$o/redfish_UpdateService_DevFirmware.json" 2>/dev/null | sed 's/.*DevFirmware\///; s/"$//'); do
      hl_fn init "redfish_devfw_$p" "$o/redfish_devfw_$p.json" 30 hl_redfish "/redfish/v1/UpdateService/DevFirmware/$p"
    done
  fi
  touch "$HL_OUT/.dmesg_mark"; return 0
}
# SDR repository cache: sel/sdr elist re-read the whole repository over KCS each time (7-14 s); with -S
# they take about 1 s. One dump (about 5 s) before the first slow read serves every later one; a light healthy run never
# dumps (no slow reads in healthy default runs). No cache: the slow full reads. Logged as an init step.
# The dump runs under sudo, so its file is root-owned: it goes outside the run tree (a copy of the run tree as
# the run user failed on it), at one fixed path (a fixed sudoers line; telemetry.py keeps its own); each run
# dumps it afresh and uses it only when a cached sdr elist gives rows (a partial dump does not). HEAVY_SDR_CACHE: another
# path (tests only).
hl_sdr_cache() {
  [ $HL_IPMI_OK = 1 ] && [ -z "$HL_SDRC" ] && [ ! -e "$HL_OUT/sdr_cache.txt" ] || return 0
  local c=${HEAVY_SDR_CACHE:-/tmp/tray-test-sdr.cache}
  hl_step init sdr_cache "$HL_OUT/sdr_cache.txt" 60 ${HL_SUDO:+$HL_SUDO} "${HEAVY_IPMITOOL:-ipmitool}" sdr dump "$c"
  # only this run's dump: a failed one may leave an old file at the fixed path (root-owned, so not removed here)
  if [ "$(tail -1 "$HL_OUT/steps.tsv" | cut -f2,3)" = "$(printf 'sdr_cache\t0')" ] && [ -s "$c" ] \
    && timeout 30 ${HL_SUDO:+$HL_SUDO} "${HEAVY_IPMITOOL:-ipmitool}" -S "$c" sdr elist 2> /dev/null | grep -q '|'; then HL_SDRC=$c
  else echo "SDR cache $c not used: the dump failed or gives no rows" >> "$HL_OUT/sdr_cache.txt"; hl_sdr_cache_drop; fi; return 0
}
hl_sdr_cache_drop() { HL_SDRC=""; return 0; }   # later reads use no -S; the root-owned file stays for the next run
hl_full() { [ "${HEAVY_FULL:-1}" = 1 ] || [ "${FC:-0}" = 1 ] || { [ -n "${FL_OUT:-}" ] && [ -f "$FL_OUT/fc_on" ]; }; }   # the slow reads: --heavy-logs, or after a fault (fl_find)
heavy_logs_snapshot() {  # <phase>
  [ -n "$HL_OUT" ] || return 0
  local ph=${1:-snap} o m=light; HL_N=$(( HL_N + 1 )); o=$HL_OUT/$(printf '%02d' $HL_N)_$ph; mkdir -p "$o"; hl_full && m=full
  { hl_ts; uptime; echo "mode: $m"; } > "$o/when.txt" 2>&1
  # BMC reads (KCS, and i2c to the tray CPLDs; no chip access) start first and run alongside the host and chip reads below
  # (31-38 s of each 45 s snapshot were serial BMC reads). In order inside the group, so CPLD debug-index
  # writes never race; joined before return, so a pre-reset snapshot still ends before the reset.
  local bp=""
  [ $m = full ] && hl_sdr_cache   # before the BMC group below: it runs in a subshell
  {
    if [ $HL_IPMI_OK = 1 ] && [ $m = full ]; then   # light: the SEL keeps its timestamped events, read after the fact
      hl_step "$ph" ipmi_sel "$o/ipmi_sel.txt" 120 ${HL_SUDO:+$HL_SUDO} "${HEAVY_IPMITOOL:-ipmitool}" ${HL_SDRC:+-S "$HL_SDRC"} sel elist
      [ -f "$HL_OUT/.sel_last" ] && diff "$HL_OUT/.sel_last" "$o/ipmi_sel.txt" | sed -n 's/^> //p' > "$o/ipmi_sel_delta.txt"
      cp "$o/ipmi_sel.txt" "$HL_OUT/.sel_last" 2>/dev/null
      hl_step "$ph" ipmi_sdr "$o/ipmi_sdr.txt" 120 ${HL_SUDO:+$HL_SUDO} "${HEAVY_IPMITOOL:-ipmitool}" ${HL_SDRC:+-S "$HL_SDRC"} sdr elist
      grep -E "$HL_SDR_RE" "$o/ipmi_sdr.txt" > "$o/sdr_power.txt" 2>/dev/null
      hl_fn "$ph" cpld_dump "$o/cpld_dump.txt" 180 hl_cpld_dump
      [ -f "$HL_ROOT/cpld_5f.py" ] && hl_step "$ph" cpld_5f "$o/cpld_5f.log" 120 python3 "$HL_ROOT/cpld_5f.py" --out "$o" \
        --ipmitool "timeout 15 ${HL_SUDO:+$HL_SUDO} ${HEAVY_IPMITOOL:-ipmitool}"
    fi
    if [ $HL_RF = 1 ] && [ $m = full ]; then
      hl_fn "$ph" redfish_power "$o/redfish_power.json" 30 hl_redfish /redfish/v1/Chassis/chassis/Power
      hl_fn "$ph" redfish_thermal "$o/redfish_thermal.json" 30 hl_redfish /redfish/v1/Chassis/chassis/Thermal
    fi
  } &
  bp=$!
  hl_fn "$ph" holders "$o/holders.txt" 30 hl_holders
  # hung chips first (lspci rev ff, ARC timeouts in the kernel log since the last snapshot): no hwmon reads of them below
  hl_step "$ph" lspci_vvv_chips "$o/lspci_vvv_chips.txt" 60 ${HL_LSUDO:+$HL_LSUDO} lspci -vvv -d 1e52:
  hl_step "$ph" journal_kernel "$o/journal_kernel.txt" 60 bash -c \
    "${HL_XSUDO:+$HL_XSUDO} journalctl -k --no-pager -o short-iso --since '@$(stat -c %Y "$HL_OUT/.dmesg_mark" 2>/dev/null || echo 0)'"
  touch "$HL_OUT/.dmesg_mark"
  hl_fn "$ph" hung_chips "$o/hung_chips.txt" 30 hl_hung "$o"
  HL_HUNG=$(awk '{ print $1 }' "$o/hung_chips.txt" 2>/dev/null | tr '\n' ' ')
  hl_hwmon_guard "$o"   # only with --hwmon-guard
  hl_fn "$ph" hwmon_modes "$o/hwmon_modes.txt" 30 hl_hwmon_modes
  [ $m = full ] && hl_fn "$ph" sysfs_telemetry "$o/sysfs_telemetry.txt" 60 hl_telem   # light: ub0_repro.py --telem has the 1 s heartbeat
  hl_fn "$ph" pcie_aer "$o/pcie_aer.txt" 60 hl_aer
  hl_fn "$ph" chips_per_tray "$o/chips_per_tray.txt" 60 hl_trays
  hl_fn "$ph" lspci_vvv_rootports "$o/lspci_vvv_rootports.txt" 120 hl_lspci_rp
  hl_step "$ph" dmesg "$o/dmesg_tail.txt" 30 bash -c "{ dmesg -T 2>/dev/null || ${HL_XSUDO:+$HL_XSUDO} dmesg -T; } | tail -n 2000"
  grep -aiE 'tenstorrent|aer|pcie|link (down|up)|surprise|hotplug|mce|machine check|hardware error|apei|ghes|i2c|smbus' \
    "$o/dmesg_tail.txt" > "$o/dmesg_excerpt.txt" 2>/dev/null
  hl_have ras-mc-ctl && hl_step "$ph" mce "$o/mce.txt" 30 ${HL_XSUDO:+$HL_XSUDO} ras-mc-ctl --errors
  if hl_dev_free; then   # these open the chips: only with nothing else on /dev/tenstorrent
    hl_have tt-smi && hl_step "$ph" tt_smi_snapshot "$o/tt_smi.txt" 120 bash -c "${HEAVY_AS:-} tt-smi -s --snapshot_no_tty 2>&1 || ${HEAVY_AS:-} tt-smi -s"
    [ -f "$HL_DIR/ethctr.py" ] && hl_fn "$ph" eth_counters "$o/eth_counters.jsonl" 180 hl_eth
  else
    echo "skipped: /dev/tenstorrent is held (see holders.txt) or HEAVY_DEVICE_READS=0" > "$o/device_reads_skipped.txt"
  fi
  wait "$bp" 2> /dev/null   # every step in the group has its own timeout
  return 0
}
heavy_logs_on_hang() {  # <bdf...> suspects (none: every chip). Before any reset.
  [ -n "$HL_OUT" ] || return 0
  declare -F mg_defer_heavy > /dev/null && mg_defer_heavy "$@" && return 0   # a chip the MCE guard removed: after its rescan
  local o b rp pairs=() ev h; h=$HL_OUT/hang_$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$h"
  { hl_ts; echo "suspects: ${*:-all}"; } > "$h/when.txt"
  for b in "$@"; do [[ $b == *:*:* ]] || b=0000:$b; rp=$(hl_chips | awk -v b="$b" '$1 == b {print $2}'); [ -n "$rp" ] && pairs+=("$rp=$b"); done
  [ ${#pairs[@]} -gt 0 ] || while read -r b rp _; do pairs+=("$rp=$b"); done < <(hl_chips)
  hl_fn hang holders "$h/holders.txt" 30 hl_holders
  # HEAVY_DEVICE_READS=0 (run.sh: a lost chip the MCE guard left on PCI): no tt-triage or exalens read
  if [ "${HEAVY_DEVICE_READS:-auto}" = 0 ]; then echo "skipped: HEAVY_DEVICE_READS=0 (no tt-triage/exalens device read)" > "$h/device_reads_skipped.txt"
    heavy_logs_snapshot hang; return 0; fi
  # noc_snapshot.sh: lspci -vvv RP+EP, dmesg, tt-triage NOC0 then NOC1 (its own steps.tsv)
  ev=("SNAP_LSPCI=timeout 10 ${HL_LSUDO:+$HL_LSUDO }lspci" "SNAP_TRIAGE_TIMEOUT_S=${HEAVY_TRIAGE_TIMEOUT_S:-120}" SNAP_DMESG_LINES=2000)
  [ -n "${HEAVY_TRIAGE:-}" ] && ev+=("SNAP_TRIAGE=$HEAVY_TRIAGE")
  hl_step hang noc_snapshot "$h/noc_snapshot.log" $(( 2 * ${HEAVY_TRIAGE_TIMEOUT_S:-120} + 300 )) \
    env "${ev[@]}" bash "$HL_ROOT/noc_snapshot.sh" "$h/noc_snapshot" "${pairs[@]}"
  if [ -n "${HEAVY_TRIAGE:-}" ]; then
    hl_step hang dump_fast_dispatch "$h/dump_fast_dispatch.txt" "${HEAVY_TRIAGE_TIMEOUT_S:-120}" bash -c \
      "$HEAVY_TRIAGE --run=dump_fast_dispatch --disable-progress --skip-version-check"
  fi
  HL_EXA_LATER=""
  if [ -f "$HL_DIR/exalens_probe.py" ] && [ -z "$(hl_holders)" ]; then hl_exalens "$h" "${pairs[@]}"
  elif [ -f "$HL_DIR/exalens_probe.py" ] && [ "${HOLD_ACTIVE:-0}" = 1 ]; then
    HL_EXA_LATER="$h ${pairs[*]}"   # --hold keeps the hung job: the probe runs once the hold ends (hold_point)
  fi
  heavy_logs_snapshot hang
  return 0
}
hl_exalens() {  # <hang dir> <rp=bdf...>: the exalens NOC0/NOC1 probe of each chip (only while nothing holds the device)
  local h=$1 b n m; shift
  for b in "$@"; do b=${b#*=}
    for n in 0 1; do hl_step hang "exalens_noc${n}_$b" "$h/exalens_noc${n}_${b//:/_}.log" 120 ${HEAVY_AS:-} \
      "${HEAVY_EXALENS_PY:-${HEAVY_PY:-python3}}" "$HL_DIR/exalens_probe.py" --noc "$n" --bdf "$b" --unique-id none \
      --out "$h/exalens_noc${n}_${b//:/_}.json"; done
    # a probe whose python lacks its module wrote only an init_error into its JSON: say so in the run log
    m=$({ cat "$h/exalens_noc0_${b//:/_}.json" "$h/exalens_noc1_${b//:/_}.json" 2> /dev/null || true; } \
      | sed -n 's/^ *"missing_module": "\([^"]*\)".*/\1/p' | head -n 1) || m=""   # no JSON (step skipped): no WARN, no abort
    [ -z "$m" ] || echo "WARN: hang capture: python module '$m' is missing from ${HEAVY_EXALENS_PY:-${HEAVY_PY:-python3}}:" \
      "no tt-exalens NOC state for $b (tray $(hl_tray "$b")); rebuild the venv with traytest/setup.sh" >&2
  done
}
# a TERM from heavy_logs_bmc_poll_stop ends a poll loop with nothing left behind. Without the trap the loop's
# subshell died at once and its sleep (or an ipmitool read in flight) ran on as an orphan, still appending rows after
# the stop; the TERM trap lets a read in flight finish (its timeout bounds it), stops the loop's own sleep by its pid
# and ends the loop. The sleep runs in the background so the trap runs at once, not after it.
hl_poll_sleep() {
  sleep "${HEAVY_POLL_S:-2}" & sp=$!
  [ $stop = 0 ] || kill "$sp" 2> /dev/null   # a TERM between the fork and sp=$! saw no sp
  wait "$sp" 2> /dev/null; sp=""
}
hl_poll_loop() {  # SDR power/temps every HEAVY_POLL_S s: "utc,sensor,value,unit,status" rows
  local f=$1 t main=$$ sp="" stop=0
  trap 'stop=1; [ -z "$sp" ] || kill "$sp" 2> /dev/null' TERM   # see hl_poll_sleep
  echo "utc,sensor,value,unit,status" > "$f"
  while [ $stop = 0 ] && kill -0 "$main" 2>/dev/null; do   # ends by itself once the shell that started it is gone
    t=$(date -u +%FT%T.%3NZ)
    timeout 20 ${HL_SUDO:+$HL_SUDO} "${HEAVY_IPMITOOL:-ipmitool}" ${HL_SDRC:+-S "$HL_SDRC"} sdr elist 2>/dev/null | grep -E "$HL_SDR_RE" |
      awk -F'|' -v t="$t" '{ gsub(/^ +| +$/, "", $1); gsub(/^ +| +$/, "", $3); gsub(/^ +| +$/, "", $5); n = split($5, v, " ")
                             printf "%s,%s,%s,%s,%s\n", t, $1, v[1], (n > 1 ? v[2] : ""), $3 }' >> "$f"
    hl_poll_sleep
  done
}
hl_pw_loop() {  # the light poll, the same rows: Get Sensor Reading 0x35-0x38 (UBB0-3, x10 W) and 0x60 (HSC, x2 W),
  # tray CPLD reg 0x19 (UBB0-3 = bus 0x11/0x13/0x0D/0x0F at 0x80; ff = all rails good), as lib/fleet.sh's fl_power_start
  local f=$1 t main=$$ n v x st b k sp="" stop=0
  trap 'stop=1; [ -z "$sp" ] || kill "$sp" 2> /dev/null' TERM   # see hl_poll_sleep
  echo "utc,sensor,value,unit,status" > "$f"
  while [ $stop = 0 ] && kill -0 "$main" 2>/dev/null; do
    t=$(date -u +%FT%T.%3NZ); k=0
    for n in 0x35:10:Power_UBB0 0x36:10:Power_UBB1 0x37:10:Power_UBB2 0x38:10:Power_UBB3 0x60:2:Power_HSC; do
      read -r v x _ <<< "$(timeout 5 ${HL_SUDO:+$HL_SUDO} "${HEAVY_IPMITOOL:-ipmitool}" raw 0x04 0x2d "${n%%:*}" 2>/dev/null)"
      if ! [[ "$v$x" =~ ^[0-9a-fA-F]{4}$ ]]; then echo "$t,${n##*:},,Watts,no answer"
      elif (( 16#$x & 0x20 )); then echo "$t,${n##*:},,Watts,na"
      else b=${n#*:}; echo "$t,${n##*:},$(( 16#$v * ${b%%:*} )),Watts,ok"; fi
    done >> "$f"
    for b in 0x11 0x13 0x0D 0x0F; do
      v=$(timeout 5 ${HL_SUDO:+$HL_SUDO} "${HEAVY_IPMITOOL:-ipmitool}" raw 0x06 0x52 "$b" 0x80 0x01 0x19 2>/dev/null | tr -d ' \n')
      st=ok; [ "${v,,}" = ff ] || st="rails ${v:-no answer}"; echo "$t,vr_pg_UBB$k,${v:-},hex,$st"; k=$((k + 1))
    done >> "$f"
    hl_poll_sleep
  done
}
heavy_logs_bmc_poll_start() {  # [label]
  [ -n "$HL_OUT" ] && [ $HL_IPMI_OK = 1 ] && [ -z "$HL_POLL_PID" ] || return 0
  local f; f=$HL_OUT/bmc_poll_$(date -u +%Y%m%dT%H%M%SZ)${1:+_$1}.csv
  if hl_full && [ -z "${BH_PID:-}" ]; then hl_sdr_cache; hl_poll_loop "$f" > /dev/null 2>&1 & HL_POLL_PID=$!   # BH_PID: lib/bmc_heat.sh polls the SDR already
  else hl_pw_loop "$f" > /dev/null 2>&1 & HL_POLL_PID=$!; fi
  printf 'poll\tbmc_poll_start\t0\t0\t%s\n' "${f#"$HL_OUT"/}" >> "$HL_OUT/steps.tsv"; return 0
}
heavy_logs_bmc_poll_stop() {  # stops only the poll this lib started, by its pid
  [ -n "$HL_POLL_PID" ] || return 0
  kill "$HL_POLL_PID" 2>/dev/null; wait "$HL_POLL_PID" 2>/dev/null; HL_POLL_PID=""; return 0
}
hl_flag() {  # $1 file (relative to HL_OUT): "ok" or "FLAG <why>"
  local f=$HL_OUT/$1
  case $1 in
    */ipmi_sel_delta.txt) [ -s "$f" ] && echo "FLAG $(wc -l < "$f") new SEL event(s)" || echo "ok (no new SEL events)"; return ;;
    */dmesg_excerpt.txt|*/journal_kernel.txt|*/holders.txt|*/hung_chips.txt|*.log) [ -s "$f" ] || { echo "ok (empty)"; return; } ;;
    bmc_poll_*.csv) awk -F, 'NR > 1 && $5 !~ /^ok$/' "$f" | grep -q . && { echo "FLAG sensor not ok"; return; }
                    [ "$(wc -l < "$f")" -gt 1 ] && echo "ok ($(( $(wc -l < "$f") - 1 )) rows)" || echo "FLAG no rows"; return ;;
  esac
  [ -s "$f" ] || { echo "FLAG empty"; return; }
  case $1 in
    */pcie_aer.txt) grep -qE 'aer_dev_(nonfatal|fatal) ' "$f" && { echo "FLAG uncorrectable AER"; return; }
                    grep -qE '^link .* x(0|1|2) ' "$f" && grep -qE 'RxErr|BadTLP|BadDLLP' "$f" && { echo "FLAG narrow link + correctable AER"; return; }
                    grep -qE 'aer_dev_correctable ' "$f" && { echo "FLAG correctable AER nonzero"; return; } ;;
    */chips_per_tray.txt) grep -q "^total ${HEAVY_WANT:-32}\$" "$f" || { echo "FLAG $(grep '^total' "$f")"; return; } ;;
    */dmesg_excerpt.txt) grep -qiE 'aer.*(error|uncorr)|surprise|link down|machine check|hardware error|ghes' "$f" && { echo "FLAG errors in dmesg"; return; } ;;
    */cpld_dump.txt) grep -E 'reg0x1E=' "$f" | grep -vqE 'reg0x1E=ff ' && { echo "FLAG a tray CPLD not alive (reg0x1E != ff)"; return; } ;;
    */cpld_reg1e_dbg5f.txt) grep -qE 'BUSBAR_SIG|HOTSWAP_OC|TRAY_12V_LOST|TRAY_OFF|CHIPS_DOWN|OTHER' "$f" && { echo "FLAG $(grep -oE 'BUSBAR_SIG|HOTSWAP_OC|TRAY_12V_LOST|TRAY_OFF|CHIPS_DOWN[A-Z0-9_]*|OTHER' "$f" | sort -u | tr '\n' ' ')"; return; } ;;
    */sysfs_telemetry.txt) local st; st=$(awk '/^== pass/{p=$3; c=$4} /tt_heartbeat|heartbeat/{h[c,p]=$2; cs[c]=1}
                    END{for (c in cs) if (h[c,1] != "" && h[c,1] == h[c,2]) n++; print n+0}' "$f")
                    [ "$st" != 0 ] && { echo "FLAG $st chip(s) heartbeat not moving"; return; } ;;
    */holders.txt) echo "FLAG /dev/tenstorrent held"; return ;;
    */hung_chips.txt) echo "FLAG $(wc -l < "$f") hung chip(s): $(awk '{ print $1 }' "$f" | tr '\n' ' ')"; return ;;
    */hwmon_modes.txt) awk '$2 == "0"' "$f" | grep -q . && { echo "FLAG hwmon guard on: $(awk '$2 == "0"' "$f" | wc -l) file(s) mode 000"; return; } ;;
    */eth_counters.jsonl) python3 -I "$HL_ROOT/fleet/fleet_py.py" ethsnap "$f" 2>/dev/null || echo "FLAG ETH counters not read"; return ;;   # FEC: run.sh's ETH FEC line
  esac
  echo ok
}
heavy_logs_finish() {  # one-page index: each artefact, its step's rc and ok/FLAG; "preempted": the steps' rc only, no checks
  [ -n "$HL_OUT" ] || return 0
  heavy_logs_bmc_poll_stop; hl_sdr_cache_drop
  local idx=$HL_OUT/index.md f rc ph st s fl
  if [ "${1:-}" = preempted ]; then   # a broker TERM: one line and the steps table, no hl_flag reads
    { echo "# heavy logs: $(hostname -s) $(hl_ts)"; echo
      echo "exit snapshot and index checks skipped: preempted (SIG${PG_DYING:-TERM})"; echo
      echo "| phase | artefact | rc | s |"; echo "|---|---|---|---|"
      awk -F'\t' 'NR > 1 { printf "| %s | %s | %s | %s |\n", $1, $5, $3, $4 }' "$HL_OUT/steps.tsv"
    } > "$idx" 2>/dev/null
    return 0
  fi
  { echo "# heavy logs: $(hostname -s) $(hl_ts)"; echo
    echo "steps: $(awk 'NR > 1' "$HL_OUT/steps.tsv" | wc -l), failed or timed out: $(awk -F'\t' 'NR > 1 && $3 != 0' "$HL_OUT/steps.tsv" | wc -l)"
    echo "skipped: $(cat "$HL_OUT"/static/*_skipped.txt "$HL_OUT"/*/device_reads_skipped.txt 2>/dev/null | sort | uniq -c | tr -s ' ' | tr '\n' ';')"; echo
    echo "| phase | artefact | rc | s | check |"; echo "|---|---|---|---|---|"
    while IFS=$'\t' read -r ph st rc s f; do
      fl=$(hl_flag "$f"); [ "$rc" = 0 ] || fl="FLAG rc $rc$([ "$rc" = 124 ] && echo ' (timed out)'); $fl"
      echo "| $ph | $f | $rc | $s | $fl |"
    done < <(awk 'NR > 1' "$HL_OUT/steps.tsv")
    for f in "$HL_OUT"/*/dmesg_excerpt.txt "$HL_OUT"/*/ipmi_sel_delta.txt "$HL_OUT"/*/cpld_reg1e_dbg5f.txt "$HL_OUT"/*/sdr_power.txt \
             "$HL_OUT"/bmc_poll_*.csv; do
      [ -e "$f" ] || continue; f=${f#"$HL_OUT"/}; echo "| ${f%%/*} | $f | - | - | $(hl_flag "$f") |"
    done
  } > "$idx" 2>/dev/null
  echo "heavy logs: $(grep -c 'FLAG' "$idx") flagged of $(awk 'NR > 1' "$HL_OUT/steps.tsv" | wc -l) steps; index $idx"
  return 0
}
