#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

declare -F pg_run > /dev/null || . "$(dirname "${BASH_SOURCE[0]}")/pgroups.sh"   # setsid children are recorded; pg_run
declare -F rk_umd > /dev/null || . "$(dirname "${BASH_SOURCE[0]}")/umd_index.sh"   # open a chip by its UMD index
# lib/fault_capture.sh: run.sh's capture on a fault, in every run: even without --heavy-logs, any
# fault gathers data to pinpoint it, without the heavy redundant logs that would run all the time
# when --heavy-logs is off. A healthy run runs none of this; it gets lib/heavy_logs.sh's light
# snapshots. Once a fault is named (FC=1) every snapshot is the full one (SEL, SDR, CPLD, Redfish). Sourced by run.sh after
# lib/heavy_logs.sh (whose read-only, time-bounded steps it uses; output under $OUT/heavy); uses run.sh's RK_MAP,
# rk_tray, rk_short, say and OUT. Never changes the run's result or its reset.
#   fault_map                       after rk_map: remember each chip's root port (a chip that falls off the bus loses it)
#   fault_name <what> <chip...>     pinpoint: chip, PCI address, root port, tray per chip, plus trays missing from
#                                   PCIe; to $OUT/fault.txt and the log
#   fault_attr <what> <0|1> <file...> a failed or hung step: the chips its own error names (fleet/fleet_py.py attr on
#                                   its log/JSON); else, for a failure (1), each chip probed alone; else UNATTRIBUTED
#                                   with every candidate. Sets FA_CHIPS, FA_TAG ("02:00.0 (tray 1) tt-smi 1, ..."), FA_WHY
#   fault_eth <log...>              an ETH core not active again at mesh open (tt-metal llrt.cpp "Timed out while waiting for
#                                   active ethernet core X-Y"): "ETH FAIL" per chip and core naming BDF, tray, core, channel/port
#                                   and link status (fleet_py.py ethcore; the newest C2/C3 tt-smi snapshot). Sets FE_LINES
#                                   and a fault_note each (FE_NONOTE=1: no notes, the caller decides)
#   fault_note <line>               one line to fault.txt and the log
#   fault_name_hang <what> <json>   a CCL hang: ub0_repro.py's hang suspects (heartbeat stood still, root port link
#                                   down) from the try's result JSON, named; their chip numbers in FC_SUS
#   fault_hang <bdf...>             a CCL hang, before the reset: NOC snapshot (lspci -vvv RP+EP, tt-triage NOC0/1),
#                                   fast-dispatch dump, exalens NOC0/1, then a snapshot (heavy_logs_on_hang)
#   fault_snapshot <phase>          a chip/tray drop, a failed try, chips missing at the start: CPLD, SEL, SDR, AER,
#                                   lspci -vvv, dmesg/journal, sysfs telemetry (heavy_logs_snapshot)
#   fault_exit                      the EXIT trap: an early exit (exit 6) after a fault still gets its last snapshot
#                                   and index.md
FC=0; FC_DONE=0; FC_MAP=""; FC_SUS=""; FA_CHIPS=""; FA_TAG=""; FA_WHY=""; FA_LINES=""; FE_LINES=""
FC_PY=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../fleet/fleet_py.py
fc_init() { [ -n "$HL_OUT" ] || heavy_logs_init "$OUT"; FC=1; }
fault_map() { FC_MAP=$(hl_chips); }
fault_bdfs() { local c; for c in "$@"; do awk -v c="$c" '$2 == c { print $1 }' <<< "$RK_MAP"; done | sort -u; }
fault_name() {  # <what> <chip...>
  local what=$1 c b rp t sh; shift
  for c in $(echo "$@" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -un); do
    b=$(awk -v c="$c" '$2 == c { print $1; exit }' <<< "$RK_MAP")
    rp=$(awk -v b="$b" '$1 == b { print $2; exit }' <<< "$FC_MAP"); t=$(rk_tray "${b:-0000:ff:00.0}")
    echo "$what: chip $c, PCI ${b:-?}, root port ${rp:-?}, tray $t, tt-smi $c"
  done > "$OUT/.fault_new"
  sh=$(rk_short); [ -n "$sh" ] && echo "$what: not on PCIe: $sh" >> "$OUT/.fault_new"
  [ -s "$OUT/.fault_new" ] || echo "$what: chip not pinpointed" > "$OUT/.fault_new"
  while read -r l; do fault_note "$l"; done < "$OUT/.fault_new"
  rm -f "$OUT/.fault_new"
}
fault_eth() {  # <log>...
  local m smi=/dev/null f l; FE_LINES=""; m=$OUT/.fe_map; echo "$RK_MAP" > "$m"
  for f in "${FL_OUT:-/nonexistent}/ttsmi_c2.json" "${FL_OUT:-/nonexistent}/ttsmi_pre.json"; do [ -s "$f" ] && { smi=$f; break; }; done
  FE_LINES=$(python3 -I "$FC_PY" ethcore "$m" "$smi" "$@" --host "$(hostname -s)" 2>/dev/null); rm -f "$m"
  while read -r l; do [ -n "$l" ] && [ -z "${FE_NONOTE:-}" ] && fault_note "$l"; done <<< "$FE_LINES"
}
fault_note() { declare -F fl_bb_dup > /dev/null && fl_bb_dup note "$*" && return 0   # a bus-bar part repeating the stage's recorded fault
  say "fault: $*"; echo "$(date '+%F %T %Z') $*" >> "$OUT/fault.txt"; }
fc_tag() {  # chip numbers -> "02:00.0 (tray 1) tt-smi 1, ..." (tt-smi's index is the tt-kmd number)
  local c b o=""
  for c in "$@"; do b=$(awk -v c="$c" '$2 == c { print $1; exit }' <<< "$RK_MAP")
    o="$o${o:+, }${b#0000:} (tray $(rk_tray "${b:-0000:ff:00.0}")) tt-smi $c"; done
  echo "$o"
}
fc_bdfk() {  # chip numbers -> "<bdf>@<chip>" per chip ("?@<chip>" if not in RK_MAP), for chunk_sched.py where
  local c b; for c in "$@"; do b=$(awk -v c="$c" '$2 == c { print $1; exit }' <<< "$RK_MAP"); echo "${b:-?}@$c"; done
}
fault_attr() {  # <what> <1 a failure, 0 a hang> <log|json>...
  local what=$1 pr=$2 m o c b u bad="" all n=0 t0 d code; shift 2; FA_CHIPS=""; FA_TAG=""; FA_WHY=""; FA_LINES=""
  m=$OUT/.fa_map   # RK_MAP plus each chip's asic id from fw_preflight's fw.txt (UMD names a chip by it)
  awk 'FILENAME == ARGV[1] { a[$1] = $3; next } NF { print $1, $2, ($1 in a ? a[$1] : "-") }' "$OUT/fw.txt" - <<< "$RK_MAP" > "$m" 2> /dev/null || echo "$RK_MAP" > "$m"
  o=$(python3 -I "$FC_PY" attr "$m" "$@" 2>/dev/null); rm -f "$m"
  if [ -n "$o" ]; then
    FA_LINES=$o   # one "<kmd> <bdf> <its error>" per chip: fl_attr files one finding each
    FA_CHIPS=$(echo "$o" | awk '{ print $1 }' | paste -sd' '); FA_WHY="its error: $(echo "$o" | head -1 | cut -d' ' -f3- | cut -c1-140)"
    FA_TAG=$(fc_tag $FA_CHIPS); fault_name "$what ($FA_WHY)" $FA_CHIPS; return 0
  fi
  all=$(echo "$RK_MAP" | awk '$2 ~ /^[0-9]+$/ { print $2 }' | sort -n | paste -sd' ')
  if [ "$pr" != 1 ]; then FA_WHY="no chip in its error; a hung chip is not probed before the reset"
  elif [ -n "${TT_VISIBLE_DEVICES+x}" ]; then FA_WHY="no chip in its error; no per-chip probe: TT_VISIBLE_DEVICES is already set"
  else   # each chip opened alone, in its own session and process group (setsid; timeout signals only that group)
    d=$OUT/probe; mkdir -p "$d"; t0=$(date +%s)
    code=${FAULT_PROBE_PY:-'import ttnn
d = ttnn.open_device(device_id=0)
ttnn.close_device(d)
print("probe ok")'}
    for c in $all; do
      [ $(( $(date +%s) - t0 )) -ge "${FAULT_PROBE_BUDGET_S:-600}" ] && break
      [ -n "${PG_DYING:-}" ] && break   # TERMed (a broker preemption): open no more chips
      # in the background and recorded (lib/pgroups.sh): a TERM's trap runs at once and reaps it
      # by its UMD index, not /dev $c: /dev 16-31 as an index opens the other tray's twin chip
      b=$(awk -v c="$c" '$2 == c { print $1; exit }' <<< "$RK_MAP"); umd_snap "$d/chip$c.log"; u=$(rk_umd "$b") || { bad="$bad $c"; n=$((n + 1)); continue; }
      setsid -w timeout -k 5 "${FAULT_PROBE_S:-60}" "${AS[@]}" env TT_VISIBLE_DEVICES="$u" "$PY" -I -c "$code" > "$d/chip$c.log" 2>&1 < /dev/null {PG_OUT}>&- &
      pg_track $!; wait $! || bad="$bad $c"; n=$((n + 1)); rk_opencheck "$d/chip$c.log" "fault probe of chip $c" "$b" || bad="$bad $c"
    done
    bad=$(echo $bad)
    if [ -n "$bad" ] && [ "$(echo $bad | wc -w)" -lt "$n" ]; then
      FA_CHIPS=$bad; FA_WHY="no chip in its error; failed a probe alone ($d)"; FA_TAG=$(fc_tag $FA_CHIPS)
      fault_name "$what ($FA_WHY)" $FA_CHIPS; return 0
    fi
    FA_WHY="no chip in its error; per-chip probe: $(echo $bad | wc -w) of $n chips failed alone ($d)"
  fi
  FA_TAG=$(fc_tag $all); fault_note "$what: UNATTRIBUTED FAIL ($FA_WHY); candidates: $FA_TAG"; return 1
}
fault_name_hang() {  # <what> <result json>: ub0_repro.py's own suspects (not the every-chip fallback)
  local s; FC_SUS=""; s=$(python3 -I - "$2" 2>/dev/null <<'PYEOF'
import json, sys
s = (json.load(open(sys.argv[1])).get("noc_snapshot") or {}).get("suspects") or {}
print(" ".join(str(c) for c, v in s.items() if (v or {}).get("why") not in (None, "all")))
PYEOF
)
  if [ -n "${s// /}" ]; then fault_name "$1 (heartbeat stall or root port link down)" $s
  else fault_name "$1 (no heartbeat stall or link down seen; $(tools_captured_txt))"; fi
  FC_SUS=$s
}
fault_hang() { fc_init; heavy_logs_on_hang "$@"; }
fault_snapshot() { fc_init; heavy_logs_snapshot "$1"; }
fault_exit() {
  declare -F bh_poll_stop > /dev/null && bh_poll_stop quiet   # lib/bmc_heat.sh's poll ends with the run
  [ -n "${HL_OUT:-}" ] || return 0
  heavy_logs_bmc_poll_stop
  [ "$FC_DONE" = 0 ] || return 0; FC_DONE=1
  if [ -n "${PG_DYING:-}" ]; then   # preempted: the broker KILLs 10 s after its TERM; no exit snapshot, no index checks
    heavy_logs_finish preempted > /dev/null 2>&1
    echo "$(date '+%F %T %Z') $(hostname -s): heavy logs: exit snapshot and index checks skipped: preempted (SIG$PG_DYING)" >&"${PG_OUT:-1}"; return 0
  fi
  { heavy_logs_snapshot exit; heavy_logs_finish; } > /dev/null 2>&1   # the run's last line stays its own
}
