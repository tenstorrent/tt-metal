# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# shellcheck shell=bash disable=SC2046,SC2010,SC2034  # word lists of BDFs; sysfs names; MG_* read by callers
# lib/mce_guard.sh: the MCE guard. Sourced by run.sh: functions only.
# Why: a process that keeps reading a hung chip's MMIO (hwmon reads go to its ARC) makes the CPU wait on a read that
# never completes; 40-180 s after the hang the host takes a fatal Bank-5 'Watchdog Timeout' MCE (F1-GBH01 10-15/10-16:
# BERT ADDR at ioread32, each after 'Timeout waiting for ARC'). A shorter PCIe completion timeout did not help (F13/F14).
# What: on a chip-hang detection in any stage (a job's hang, its HANG file, a chip lost from the load's watch, a try past
# its wall-clock limit), when a hung chip reads ffffffff in config space or the kernel logged 'Timeout waiting for ARC
# response' for it, the guard, in this order, inside MG_BUDGET_S (30 s) from the detection:
#   1. stops this run's own load (the caller's stop command, this run's process groups only) and this run's own chip
#      reader (C8 telemetry, FL_C8_PID); nothing else is signalled (tt-telemetry, exporters, fabric agent keep running)
#   2. captures triage (FLEET_ON_HANG / on_hang.sh) under a hard cap: what is left of the budget minus MG_REMOVE_S
#   3. re-checks the gates (below) and removes each hung chip from the PCI bus (sysfs remove): a reader left
#      anywhere then gets ENODEV instead of a stalled CPU core
#   and logs each step with its seconds since the first hang detection (mce_guard/timeline.tsv) by BDF and tray 1-4.
# Timing: the budget and every timeline second count from the FIRST detection the caller passes (a job's
# beat or wall-clock limit, its HANG file's first line, a try's FIRST LOSS line), not from the guard's own check, which
# can come after a 180 s capture; the guard's lines print both ("N s after the first detection, M s after the guard's
# check"), and a cut 40 s or more after the first detection is a WARN.
# No serving or neighbour snapshot: the guard reads only this host's own chips, root ports and
# device holders; nothing it does waits on, or stops for, another host.
# AER storm (seen on one box: 02:00.0 (tray 1) at config ffff, its root port 00:01.2 logging 72-108 AER
# records a minute for 55 min, then a fatal Bank-5 MCE): a chip whose root port logged MG_AER_MIN (30) or more kernel
# AER lines in the last MG_AER_WIN_S (60 s) also counts as hung. These hosts report AER firmware-first, and a sysfs
# remove changes nothing in the port's hardware or firmware reporting, so after the remove of a chip whose port storms
# (and has no other chip below it) the guard holds that port's secondary bus reset (BRIDGE_CONTROL bit 6, that reset
# held: the link stays down, so no request can wait on it) until mg_recover releases it (the saved value written back,
# then the rescan); mg_report releases any port still held. MG_AER_HOLD=0 turns the hold off.
# After the stage (rk_end_, before its reset): mg_recover rescans each removed chip's root port (a secondary bus reset
# of that root port first when the rescan alone does not bring it back), then the usual reset path runs as
# always (warm reset, rescan, tray re-power) and recounts.
# Gates (the guard acts only where run.sh's own resets may run): armed (mg_arm) only when MG_ON (TRAY_REPRO_MCE_GUARD,
# default 1; --no-mce-guard sets 0), root works (rk_can_root) and, in --mode runs, C2's start reset was not skipped by
# its gate; at each hang, before anything is stopped, the reset gate must pass (rk_gate: TRAY_REPRO_RESET_GATE, this
# host's own serving check), else it disarms and the old path runs. Before each remove it checks
# again: the reset gate under MG_GATE_S, no FL_STOP, and no process of
# another login user (not this run's user, not root) holding /dev/tenstorrent. A refused gate: nothing is removed,
# the old path runs. Never a power cycle, never a TDP change.
# env: MG_NOCAP (1: the caller already captured), TRAY_REPRO_MCE_GUARD (1), MG_BUDGET_S (30), MG_REMOVE_S (6, kept for the gate and remove), MG_GATE_S (8),
#      MG_KLOG (kernel log command; default journalctl -k since the run start), MG_SBR (1: secondary bus reset in recovery),
#      MG_AERLOG (AER line source; default journalctl -k), MG_AER_MIN (30), MG_AER_WIN_S (60), MG_AER_HOLD (1)
# --hold (lib/hold.sh, HOLD_ACTIVE=1; the mode for debugging hangs
# afterwards): the guard stands down. It never PCI-removes, resets or rescans a hung chip; it stops
# run.sh's own C8 reader, marks the chip hung for the heavy-logs hwmon reads (MG_HUNG), prints one line per chip by BDF
# and tray 1-4 and returns 1, so the hold keeps the hang for tt-triage/exalens as before.
# --heavy-logs: the capped capture above (tt-triage, the only capture that reads the hung chip) ends before the cut;
# heavy_logs_on_hang for a chip the guard removed (exalens, fast-dispatch dump, BMC/SEL/SDR snapshot) waits for
# mg_recover and runs after the root port rescan as a snapshot with no device open and no hwmon read of that chip.
MG_ON=${TRAY_REPRO_MCE_GUARD:-1}; MG_ARMED=0; MG_WHY="not armed yet"; MG_T0=$(date +%s); export MG_T0; MG_KWARN=0; MG_REMOVED=(); MG_LOG=()
MG_HELD=()   # ports held in reset "rp|saved|epoch|bdf"
MG_HUNG=""; MG_HEAVY_LATER=()   # chips the guard saw hung (no hwmon reads of them); heavy captures deferred past the rescan
mg_dir() { local d=${OUT:-/tmp}/mce_guard; mkdir -p "$d" 2>/dev/null; echo "$d"; }
mg_tag() { echo "${1#0000:} (tray $(rk_tray "$1" | cut -d' ' -f1))"; }   # bdf -> "02:00.0 (tray 1)": tray 1-4, never UBB
mg_el() { echo $(( $(date +%s) - MG_DT0 )); }   # seconds since the first hang detection (the caller's epoch)
mg_elc() { echo $(( $(date +%s) - MG_CK )); }   # seconds since the guard's own check
mg_both() { echo "$(mg_el) s after the first detection ($(mg_elc) s after the guard's check)"; }
mg_rto() { local s=$1; shift; if [ "$(id -u)" = 0 ]; then timeout -k 2 "$s" "$@"; else timeout -k 2 "$s" sudo -n "$@"; fi; }   # rk_root under a timeout
mg_say() { MG_LOG+=("$*"); say "mce guard: $*"; }
mg_arm() {  # after the start reset (rk_start / fl_start's C2): arm only where this run's resets may run
  MG_ARMED=0; MG_T0=$(date +%s)   # ARC timeouts of the start reset are not this run's hangs
  if [ "$MG_ON" != 1 ]; then MG_WHY="off (--no-mce-guard / TRAY_REPRO_MCE_GUARD=0)"
  elif ! rk_can_root; then MG_WHY="not armed: no root (sudo -n), so no reset or PCI remove can run"
  elif [ -n "${FL_STOP:-}" ]; then MG_WHY="not armed: $FL_STOP"
  elif printf '%s\n' "${FL_RESET_SKIPPED[@]}" | grep -q '^C2 '; then MG_WHY="not armed: C2's start reset was skipped by its gate"
  else MG_ARMED=1; MG_WHY="armed (this run's resets may run here; the reset gate is checked at each hang)"; fi
  say "mce guard: $MG_WHY"
}
mg_dead() {  # <bdf>...: the chips that are hung for MMIO: config space reads ffffffff, or the kernel logged an ARC
             # timeout for them since the run start. Config space and the kernel log only: never a BAR or hwmon read
  local b w k="" kc=${MG_KLOG:-}
  # the host's kernel log only for the host's own /sys (a test's fake /sys reads no real log unless MG_KLOG is set)
  [ -z "$kc" ] && [ "${SYS:-/sys}" = /sys ] && kc="timeout 5 journalctl -k --no-pager -o cat --since @$MG_T0"
  if [ -n "$kc" ]; then k=$($kc 2>&1); if [ $? -ne 0 ] && [ "$MG_KWARN" = 0 ]; then MG_KWARN=1; mg_say "MCE GUARD: kernel log unreadable, ARC-timeout detection off"; k=""; fi
    k=$(grep -a 'Timeout waiting for ARC' <<< "$k"); fi
  for b in "$@"; do [ -e "$SYS/bus/pci/devices/$b" ] || continue
    w=$(timeout 2 od -An -tx4 -N4 "$SYS/bus/pci/devices/$b/config" 2>/dev/null | tr -d ' ')
    # MG_LOST=1 (mg_loss): the caller's chip check found it lost while it is still on PCI; it counts as is
    if [ "${MG_LOST:-0}" = 1 ] || [ "$w" = ffffffff ] || grep -aqF "${b#0000:}" <<< "$k" || mg_storm "$b"; then echo "$b"; fi
  done
}
mg_aer_n() {  # <root port> <since epoch>: the kernel's AER lines naming that port since then (kernel log only)
  local kc=${MG_AERLOG:-}
  [ -z "$kc" ] && [ "${SYS:-/sys}" = /sys ] && kc="timeout 5 journalctl -k --no-pager -o cat --since @$2"
  [ -n "$kc" ] || { echo 0; return 0; }
  $kc 2>/dev/null | grep -a 'AER' | grep -acF "${1#0000:}"
}
mg_storm() {  # <bdf>: 0 when the chip's root port logged MG_AER_MIN (30) or more AER lines in the last MG_AER_WIN_S (60 s)
  local rp; rp=$(rk_rp "$1"); [ -n "$rp" ] || return 1
  [ "$(mg_aer_n "$rp" $(( $(date +%s) - ${MG_AER_WIN_S:-60} )))" -ge "${MG_AER_MIN:-30}" ]
}
mg_iso_path() { echo "${OUT:-/tmp}/mce_guard/isolated.txt"; }   # the chips the guard cut this run, one BDF a line
mg_isolated() { grep -qsxF "$1" "$(mg_iso_path)"; }   # <bdf>: 0 when the guard cut it: no reader opens it again this run
mg_jreal() { [ "${SYS:-/sys}" = /sys ] || [ -n "${MG_JCTL:-}" ]; }   # mg_watch and mg_preload read the real kernel log
# (MG_JCTL: a journalctl stand-in taking the same --since/--until arguments, tests only)
mg_watch() {  # <bdf>...: fl_watch's in-run monitor: prints "<bdf> <why>" for each
              # chip still on PCI with a burst of kmd 'Failed to set initial power state' -5/-22 (MG_KMD_MIN 5 in MG_KMD_WIN_S
              # 60 s) or an AER storm on its root port (MG_AER_MIN in MG_AER_WIN_S, as mg_storm). Kernel log only, one read of
              # each per call. Nothing when the guard is unarmed or under --hold (the hang stays for debugging)
              # MG_LOAD_T0 (the run's first load start, FL_MG_T0): the window starts no earlier, so a burst before the run's load is not
              # its fault (mg_preload reports that one)
  [ "${MG_ARMED:-0}" = 1 ] && [ "${HOLD_ACTIVE:-0}" != 1 ] || return 0
  local now ks as; now=$(date +%s); ks=$(( now - ${MG_KMD_WIN_S:-60} )); as=$(( now - ${MG_AER_WIN_S:-60} ))
  [ -n "${MG_LOAD_T0:-}" ] && { [ "$ks" -lt "$MG_LOAD_T0" ] && ks=$MG_LOAD_T0; [ "$as" -lt "$MG_LOAD_T0" ] && as=$MG_LOAD_T0; }
  local kc=${MG_KMDLOG:-} ac=${MG_AERLOG:-}
  [ -z "$kc" ] && mg_jreal && kc="timeout 5 ${MG_JCTL:-journalctl} -k --no-pager -o cat --since @$ks"
  [ -z "$ac" ] && mg_jreal && ac="timeout 5 ${MG_JCTL:-journalctl} -k --no-pager -o cat --since @$as"
  mg_scan "$kc" "$ac" "$@"
}
mg_preload() {  # <load start epoch> <bdf>...: the same check over the MG_KMD_WIN_S / MG_AER_WIN_S before the load started
                # fl_watch runs it once a run and prints each hit as one WARN line,
                # pre-existing, not from this run, that never sets the verdict. Nothing unarmed or under --hold, as mg_watch
  [ "${MG_ARMED:-0}" = 1 ] && [ "${HOLD_ACTIVE:-0}" != 1 ] || return 0
  local t=$1 kc=${MG_KMDLOG_PRE:-} ac=${MG_AERLOG_PRE:-}; shift
  [ -z "$kc" ] && mg_jreal && kc="timeout 5 ${MG_JCTL:-journalctl} -k --no-pager -o cat --since @$(( t - ${MG_KMD_WIN_S:-60} )) --until @$t"
  [ -z "$ac" ] && mg_jreal && ac="timeout 5 ${MG_JCTL:-journalctl} -k --no-pager -o cat --since @$(( t - ${MG_AER_WIN_S:-60} )) --until @$t"
  mg_scan "$kc" "$ac" "$@"
}
mg_scan() {  # <kmd log cmd> <AER log cmd> <bdf>...: mg_watch's and mg_preload's per-chip count
  local kc=$1 ac=$2 b rp n k="" a=""; shift 2
  [ $# -gt 0 ] || set -- $(echo "${RK_MAP:-}" | awk 'NF { print $1 }')
  [ -n "$kc" ] && k=$($kc 2>/dev/null | grep -aE 'Failed to set initial power state: -(5|22)([^0-9]|$)')
  [ -n "$ac" ] && a=$($ac 2>/dev/null | grep -a 'AER')
  for b in "$@"; do [[ $b == *:*:* ]] || b=0000:$b
    [ -e "$SYS/bus/pci/devices/$b" ] || continue
    [[ " ${MG_REMOVED[*]:-} " == *" $b|"* ]] && continue
    n=$(grep -acF "${b#0000:}" <<< "$k")
    if [ "$n" -ge "${MG_KMD_MIN:-5}" ]; then echo "$b kmd burst: $n 'Failed to set initial power state' (-5/-22) in ${MG_KMD_WIN_S:-60} s"; continue; fi
    rp=$(rk_rp "$b"); [ -n "$rp" ] && [ -n "$a" ] || continue
    n=$(grep -acF "${rp#0000:}" <<< "$a")
    [ "$n" -ge "${MG_AER_MIN:-30}" ] && echo "$b AER storm: $n AER lines on root port ${rp#0000:} in ${MG_AER_WIN_S:-60} s"
  done
  return 0
}
mg_link_wait() {  # <root port>: up to 10 s for its link (Data Link Layer Link Active)
  local k; for k in 1 2 3 4 5 6 7 8 9 10; do
    [ "$(( 16#$(rk_root setpci -s "$1" CAP_EXP+0x12.w 2>/dev/null || echo 0) & 0x2000 ))" != 0 ] && return 0; sleep 1; done; return 1
}
mg_port_hold() {  # <stage> <bdf> <root port>: after the remove of a chip whose port storms with AER and has no other chip
                  # below it: hold that port's secondary bus reset (link down) until mg_recover; the saved value is the undo
  local st=$1 b=$2 rp=$3 bc
  [ "${MG_AER_HOLD:-1}" = 1 ] && [ -n "$rp" ] && mg_storm "$b" || return 0
  [ "$(ls -d "$SYS/bus/pci/devices/$rp"/0000:* 2>/dev/null | grep -vc "/$b\$")" = 0 ] ||
    { mg_say "$st: root port $rp of $(mg_tag "$b") storms with AER but has other chips below it: no hold"; return 0; }
  if ! bc=$(rk_root setpci -s "$rp" BRIDGE_CONTROL 2>/dev/null) || [ -z "$bc" ] ||
     ! rk_root setpci -s "$rp" BRIDGE_CONTROL="$(printf '%04x' $(( 16#$bc | 0x40 )))"; then
    mg_say "$st: AER storm on root port $rp of $(mg_tag "$b"): holding its secondary bus reset failed"; return 0; fi
  MG_HELD+=("$rp|$bc|$(date +%s)|$b")
  printf '%s\tport-held\t%s\t%s\n' "$st" "$(mg_el)" "$(mg_tag "$b"): root port $rp in secondary bus reset (undo: setpci -s $rp BRIDGE_CONTROL=$bc)" >> "$(mg_dir)/timeline.tsv"
  mg_say "$st: AER storm on root port $rp of $(mg_tag "$b") ($(mg_aer_n "$rp" $(( $(date +%s) - ${MG_AER_WIN_S:-60} ))) AER lines in ${MG_AER_WIN_S:-60} s): its secondary bus reset is held until the recovery (undo: setpci -s $rp BRIDGE_CONTROL=$bc)"
}
mg_release() {  # release every root port the guard holds in reset: the saved BRIDGE_CONTROL written back, then its link
  local e rp bc t b n
  [ ${#MG_HELD[@]} -gt 0 ] || return 0
  for e in "${MG_HELD[@]}"; do IFS='|' read -r rp bc t b <<< "$e"
    n=$(mg_aer_n "$rp" "$t")
    if rk_root setpci -s "$rp" BRIDGE_CONTROL="$bc"; then mg_link_wait "$rp"
      mg_say "recovery: released the secondary bus reset held on root port $rp of $(mg_tag "$b") after $(( $(date +%s) - t )) s; $n AER lines on it while held"
    else mg_say "recovery: WARN releasing root port $rp of $(mg_tag "$b") failed: run setpci -s $rp BRIDGE_CONTROL=$bc"
      # WARN by design: the chip is already counted by its cut or loss line; this judges the guard's recovery
      declare -F fl_find > /dev/null && fl_find "MCE GUARD WARN root port $rp of $(mg_tag "$b") still held in reset: run setpci -s $rp BRIDGE_CONTROL=$bc"; fi
  done
  MG_HELD=()
}
mg_foreign() {  # holders of /dev/tenstorrent run by another login user (not this run's user, not root): "<comm> <pid> <user>"
  local me=${HEAVY_RUN_USER:-${SUDO_USER:-$(id -un)}}
  declare -F hl_scan > /dev/null || return 0
  hl_scan | awk -v me="$me" '$3 != me && $3 != "root" { print $1, $2, $3 }' | sort -u
}
mg_gate() {  # 0 when a remove may run now; else MG_GWHY
  local f; MG_GWHY=""
  [ -n "${FL_STOP:-}" ] && { MG_GWHY="$FL_STOP"; return 1; }
  rk_can_root || { MG_GWHY="no root"; return 1; }
  if [ -n "${TRAY_REPRO_RESET_GATE:-}" ] && ! timeout -k 2 "${MG_GATE_S:-8}" bash -c "$TRAY_REPRO_RESET_GATE" > /dev/null 2>&1; then
    MG_GWHY="reset gate refused (or did not answer in ${MG_GATE_S:-8} s)"; return 1; fi
  f=$(mg_foreign); [ -n "$f" ] && { MG_GWHY="another user holds /dev/tenstorrent: $(echo "$f" | head -3 | paste -sd';')"; return 1; }
  return 0
}
mg_pre_cut() {  # <stage> <bdf> <root port>: lspci -vvv of the chip and its root port, and its AER, before the remove
  local d; d=$(mg_dir)/$1; mkdir -p "$d"
  { echo "== $(mg_tag "$2") (root port ${3:-?})"; mg_rto 10 lspci -vvv -s "$2"; [ -n "$3" ] && mg_rto 10 lspci -vvv -s "$3"; } >> "$d/pre_cut_lspci.txt" 2>&1
  python3 -I "${FL_DIR:-./fleet}/fleet_py.py" aer "$SYS" "$2" >> "$d/pre_cut_aer.tsv" 2>&1
}
mg_gone() {  # <bdf> <root port>: 0 when the chip's link is down (root port link-active clear) or its config reads ffffffff
  local w; w=$(timeout 2 od -An -tx4 -N4 "$SYS/bus/pci/devices/$1/config" 2>/dev/null | tr -d ' ')
  [ "$w" = ffffffff ] && return 0
  [ -n "$2" ] && [ "$(( 16#$(rk_root setpci -s "$2" CAP_EXP+0x12.w 2>/dev/null || echo 2000) & 0x2000 ))" = 0 ] && return 0
  return 1
}
# the cut line's level: a chip that hung for MMIO (C5 pair hang, config ffffffff) is a counted
# fault that only this line names, so it is 'FAIL MCE GUARD' (FAIL tray N), as main's 'WARN MCE GUARD' line was counted.
# A loss (MG_LOST, MG_TRYLOSS: fl_lost or the try already counts the drop) or the monitor (MG_WHAT: fleet.sh writes its own
# FAIL MCE GUARD line) stays one WARN, so no chip is counted twice.
mg_cut_lvl() {
  if [ "${MG_LOST:-0}" = 1 ] || [ "${MG_TRYLOSS:-0}" = 1 ] || [ -n "${MG_WHAT:-}" ]; then echo "MCE GUARD WARN"; else echo "FAIL MCE GUARD"; fi
}
mg_hang() {  # <stage> <first detection epoch> <stop command or ""> <bdf>...: 0 when the guard acted (load stopped,
             # capture, remove), 1 when it did not (unarmed, or no chip reads hung): the caller's old path then runs unchanged
  local st=$1 stop=$3 d tl dead b rp cap el what="hung for MMIO"; MG_CK=$(date +%s); MG_DT0=$2; shift 3
  [[ $MG_DT0 =~ ^[0-9]+$ ]] && [ "$MG_DT0" -le "$MG_CK" ] || MG_DT0=$MG_CK
  [ "$MG_ARMED" = 1 ] || return 1
  [ $# -gt 0 ] || set -- $(echo "${RK_MAP:-}" | awk 'NF { print $1 }')
  dead=$(mg_dead "$@"); [ -n "$dead" ] || return 1
  [ "${MG_LOST:-0}" = 1 ] && what="lost from PCIe (run.sh's chip check)"
  [ "${MG_TRYLOSS:-0}" = 1 ] && what="lost in the try's FIRST LOSS"
  [ -n "${MG_WHAT:-}" ] && what=$MG_WHAT   # mg_watch's trigger
  if [ "${HOLD_ACTIVE:-0}" = 1 ]; then mg_hold "$st" $dead; return 1; fi   # --hold: stand down, the hang stays for debugging
  # the reset gate (serving check) right at the hang, before anything is stopped: refused, the old path runs unchanged
  # (not at arm time: the start reset just ran it, and an extra call there shifts every later gate answer)
  rk_gate || { MG_ARMED=0; MG_WHY="not armed: $RK_GATE_WHY"; say "mce guard: $st: $MG_WHY"; return 1; }
  d=$(mg_dir); tl=$d/timeline.tsv; [ -s "$tl" ] || printf 'stage\tstep\tseconds_after_detection\tchips\n' > "$tl"
  local tags; tags=$(for b in $dead; do mg_tag "$b"; done | paste -sd';' | sed 's/;/; /g')
  printf '%s\tdetect\t0\t%s\n' "$st" "$tags" >> "$tl"
  printf '%s\tchecked\t%s\t%s\n' "$st" "$(mg_el)" "the guard's check" >> "$tl"
  mg_say "$st: $what: $tags (first detection $(date -d "@$MG_DT0" '+%T %Z'), guard check $(mg_el) s later); stopping this run's load and readers"
  declare -F th_stop > /dev/null && [ -n "${TH_PID:-}" ] && th_stop now > /dev/null 2>&1   # the thermal sampler: first
  MG_PRE_STOP=$(mg_el)
  [ -n "$stop" ] && eval "$stop" > /dev/null 2>&1
  declare -F fl_c8_stop > /dev/null && [ -n "${FL_C8_PID:-}" ] && fl_c8_stop > /dev/null 2>&1
  printf '%s\tstopped\t%s\t-\n' "$st" "$(mg_el)" >> "$tl"
  # the budget counts from after the stop: a slow stop (try_stop waits up to ~40 s) must not skip the capture
  cap=$(( ${MG_BUDGET_S:-30} - ${MG_REMOVE_S:-6} - MG_PRE_STOP ))
  if [ "${MG_NOCAP:-0}" = 1 ]; then printf '%s\tcapture-done\t%s\tcaller captured lspci/AER before the cut (mg_pre_cut)\n' "$st" "$(mg_el)" >> "$tl"
  elif [ "$cap" -ge 3 ]; then
    FLEET_HANG_DIR=$d/$st FLEET_HANG_BDFS="$dead" FLEET_HANG_CAP_S=$cap timeout -k 2 "$cap" bash "${FLEET_ON_HANG:-${FL_DIR:-./fleet}/on_hang.sh}" > /dev/null 2>&1
    printf '%s\tcaptured\t%s\tcap %s s (%s)\n' "$st" "$(mg_el)" "$cap" "$d/$st" >> "$tl"
  else printf '%s\tcapture-skipped\t%s\tno time left (cap %s s)\n' "$st" "$(mg_el)" "$cap" >> "$tl"; fi
  if ! mg_gate; then
    printf '%s\tremove-refused\t%s\t%s\n' "$st" "$(mg_el)" "$MG_GWHY" >> "$tl"
    mg_say "$st: no PCI remove of $tags: $MG_GWHY"; return 0
  fi
  for b in $dead; do
    rp=$(rk_rp "$b")
    mg_pre_cut "$st" "$b" "$rp"
    # a chip lost in a stage (mg_loss) is removed even with its link down: one run left such a chip
    # on PCI, the exalens/tt-triage capture opened it and the host took an MCE
    if [ "$st" = loss ] && [ "${MG_LOST:-0}" != 1 ] && mg_gone "$b" "$rp"; then
      printf '%s\tremove-skipped\t%s\t%s\n' "$st" "$(mg_el)" "$(mg_tag "$b")" >> "$tl"
      mg_say "$st: no PCI remove of $(mg_tag "$b"): link down or config reads ffff, removing gains nothing"
      # a gone chip whose root port storms with AER (seen: 02:00.0 at ffff, 55 min storm, then the MCE): hold
      # that port once, no remove
      [[ "${MG_HELD[*]:-}" == *"$rp|"* ]] || mg_port_hold "$st" "$b" "$rp"
      continue; fi
    if [ -e "$SYS/bus/pci/devices/$b" ] && echo 1 | mg_rto 5 tee "$SYS/bus/pci/devices/$b/remove" > /dev/null 2>&1; then
      MG_REMOVED+=("$b|$rp|$st|$(date +%s)")
      # isolated for the rest of the run: C8, the vcore sampler, heavy-logs hwmon and the loss capture skip it
      echo "$b" >> "$(mg_iso_path)"; [[ " $MG_HUNG " == *" $b "* ]] || MG_HUNG="${MG_HUNG:+$MG_HUNG }$b"
      printf '%s\tremoved\t%s\t%s\n' "$st" "$(mg_el)" "$(mg_tag "$b")" >> "$tl"
      mg_say "$st: removed $(mg_tag "$b") from PCI $(mg_both) (root port ${rp:-?}); recovery after the stage"
      declare -F fl_find > /dev/null && fl_find "$(mg_cut_lvl) $(mg_tag "$b"): $what in $st, removed from PCI $(mg_both) (mce_guard/timeline.tsv)"
      mg_port_hold "$st" "$b" "$rp"
    else printf '%s\tremove-failed\t%s\t%s\n' "$st" "$(mg_el)" "$(mg_tag "$b")" >> "$tl"
      mg_say "$st: PCI remove of $(mg_tag "$b") failed"; fi
  done
  el=$(mg_el); [ "$el" -ge 40 ] && mg_say "$st: WARN the cut came $(mg_both): over 40 s"
  return 0
}
mg_loss() {  # <stage> <bdf>...: chips run.sh's chip check found lost after a load phase (C5, a load stage, a try), after
            # the caller's own lspci/AER/SEL capture: the guard's cut (load and readers stopped, the chip PCI-removed, its
            # heavy exalens/tt-triage reads deferred past the rescan) before anything reads it; --hold stands down
  local st; st=$(echo "$1" | tr -c 'A-Za-z0-9\n' _); shift
  [ $# -gt 0 ] || return 1
  MG_LOST=1 MG_NOCAP=1 mg_hang "$st" "$(date +%s)" "" "$@"
}
mg_loss_reads() {  # <bdf>...: after mg_loss: 0 when run.sh's loss capture may open the chips (--hold, guard off, or no lost chip is
                  # left on PCI); 1, with one line naming each lost chip still on PCI (the guard did not cut it: unarmed,
                  # gate refused, remove failed): the capture then runs no tt-smi, eth counter, tt-triage or exalens read
                  # (in two runs those reads of a lost chip took the host down with an MCE)
  local b left=""
  [ "${HOLD_ACTIVE:-0}" = 1 ] && return 0
  [ "$MG_ON" = 1 ] || return 0   # --no-mce-guard: the capture reads as before the guard
  # a chip the guard removed counts as cut even if its sysfs entry lingers (the remove was written)
  for b in "$@"; do [[ $b == *:*:* ]] || b=0000:$b; [[ " ${MG_REMOVED[*]:-} " == *" $b|"* ]] && continue
    [ -e "$SYS/bus/pci/devices/$b" ] && left+="${left:+; }$(mg_tag "$b")"; done
  # a chip the guard cut earlier in this run and a rescan brought back: these reads open every chip, so none runs
  while read -r b; do [ -n "$b" ] && [ -e "$SYS/bus/pci/devices/$b" ] && [[ " ${MG_REMOVED[*]:-} " != *" $b|"* ]] &&
    [[ "$left" != *"$(mg_tag "$b")"* ]] && left+="${left:+; }$(mg_tag "$b") (isolated earlier)"; done < <(sort -u "$(mg_iso_path)" 2>/dev/null)
  [ -z "$left" ] && return 0
  mg_say "loss capture: $left lost but still on PCI (the guard did not remove it: $MG_WHY); no device read (tt-smi, eth counters, tt-triage, exalens) in this capture"
  return 1
}
mg_mce_after() {  # <bdf> <epoch>: did an MCE still follow the guard's cut of that chip: one line, by BDF and tray 1-4
  local kc=${MG_MCELOG:-} l n rc=0 out ln ev errf re='^[A-Z][a-z]{2} [ 0-9][0-9] [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+ '
  if [ -z "$kc" ] && [ "${SYS:-/sys}" = /sys ]; then
    kc="timeout 5 journalctl -k --no-pager -o short-precise --since @$2"
    timeout -k 2 5 sudo -n -l journalctl -k --no-pager -o short-precise --since "@$2" < /dev/null > /dev/null 2>&1 \
      && kc="timeout 5 sudo -n journalctl -k --no-pager -o short-precise --since @$2"
  fi
  if [ -z "$kc" ]; then mg_say "recovery: MCE after the guard's cut of $(mg_tag "$1"): not checked (no kernel log)"; return 0; fi
  errf=$(mktemp)   # its own stderr only: log text never decides "not readable"
  out=$($kc 2> "$errf") || rc=$?
  if [ "$rc" != 0 ] || grep -qaiE 'permission denied|not permitted|no journal files|not seeing messages' "$errf"; then
    rm -f "$errf"; mg_say "recovery: MCE after the guard's cut of $(mg_tag "$1"): not checked (kernel log not readable)"; return 0; fi
  rm -f "$errf"
  l=$(grep -aiE 'mce: \[hardware error\]|machine check' <<< "$out") || true
  n=$(grep -c . <<< "$l")
  if [ "$n" = 0 ]; then mg_say "recovery: MCE after the guard's cut of $(mg_tag "$1"): none in the kernel log"
  else mg_say "recovery: MCE after the guard's cut of $(mg_tag "$1"): $n kernel line(s), first: $(head -1 <<< "$l" | cut -c1-200)"
    # one WARN per timestamped kernel event, not per line or per removed chip: the lines of one x86 MCE (CPU/Bank, RIP,
    # TSC/ADDR/MISC, PROCESSOR) share one short-precise timestamp; equal texts at different times count apart; a line
    # with no timestamp is its own event
    while IFS= read -r ln; do
      [ -n "$ln" ] || continue
      ev=$ln; [[ $ln =~ $re ]] && ev=${BASH_REMATCH[0]}
      [[ $'\n'"${MG_MCE_WARNED:-}"$'\n' == *$'\n'"$ev"$'\n'* ]] && continue
      MG_MCE_WARNED+="${MG_MCE_WARNED:+$'\n'}$ev"
      # WARN by design: the chip is already counted by its cut or loss line; this judges whether the guard helped
      declare -F fl_find > /dev/null && fl_find "MCE GUARD WARN MCE still followed the PCI remove of $(mg_tag "$1"): $(cut -c1-200 <<< "$ln")"
    done <<< "$l"
  fi
}
mg_hold() {  # <stage> <bdf>...: --hold: stand down. No remove, reset or rescan; stop run.sh's own reader (C8) and keep
            # the heavy-logs hwmon reads off the chips; one line per chip, by BDF and tray 1-4
  local st=$1 b d; shift
  declare -F th_stop > /dev/null && [ -n "${TH_PID:-}" ] && th_stop now > /dev/null 2>&1   # the thermal sampler
  declare -F fl_c8_stop > /dev/null && [ -n "${FL_C8_PID:-}" ] && fl_c8_stop > /dev/null 2>&1
  for b in "$@"; do
    [[ " $MG_HUNG " == *" $b "* ]] && continue
    MG_HUNG="${MG_HUNG:+$MG_HUNG }$b"
    d=$(mg_dir); [ -s "$d/timeline.tsv" ] || printf 'stage\tstep\tseconds_after_detection\tchips\n' > "$d/timeline.tsv"
    printf '%s\theld\t%s\t%s\n' "$st" "$(mg_el)" "$(mg_tag "$b")" >> "$d/timeline.tsv"
    mg_say "$st: stood down (--hold): $(mg_tag "$b") is hung and held for debugging (tt-triage/exalens); no PCI remove, reset or rescan by the guard; run.sh's own readers of it stopped"
  done
  MG_WHY="stood down (--hold): held $(for b in $MG_HUNG; do mg_tag "$b"; done | paste -sd';' | sed 's/;/; /g')"
}
mg_defer_heavy() {  # <bdf>... (heavy_logs_on_hang's suspects; none: every chip): 0 when a suspect is a chip the guard
                    # removed and not yet rescanned: the call is queued for mg_recover, after the rescan
  local e b hit=""
  [ ${#MG_REMOVED[@]} -gt 0 ] || return 1
  for e in "${MG_REMOVED[@]}"; do b=${e%%|*}
    if [ $# = 0 ] || [[ " $* " == *" $b "* ]] || [[ " $* " == *" ${b#0000:} "* ]]; then hit+="${hit:+; }$(mg_tag "$b")"; fi
  done
  [ -n "$hit" ] || return 1
  MG_HEAVY_LATER+=("${*:-all}"); mg_say "heavy-logs hang capture of $hit waits for the root port rescan (chip off PCI)"
  return 0
}
mg_sbr() {  # <root port>: secondary bus reset of this host's own root port (treated as a tt-smi reset)
  local bc; bc=$(rk_root setpci -s "$1" BRIDGE_CONTROL 2>/dev/null) || return 1
  rk_root setpci -s "$1" BRIDGE_CONTROL="$(printf '%04x' $(( 16#$bc | 0x40 )))" && sleep 0.2 &&
    rk_root setpci -s "$1" BRIDGE_CONTROL="$bc" && sleep 1
}
mg_recover() {  # in rk_end_, before its snapshot and reset: bring back each chip the guard removed (root port rescan,
                # else a secondary bus reset of its root port when that port has only this chip below it, then rescan);
                # the goal-17 reset that follows warm-resets every chip and re-powers a tray still missing chips
  local e b rp st n=0 back=0 k
  [ ${#MG_REMOVED[@]} -gt 0 ] || { mg_release; return 0; }
  mg_release   # a port held in reset (AER storm): released before its rescan
  for e in "${MG_REMOVED[@]}"; do IFS='|' read -r b rp st k <<< "$e"; n=$((n + 1))
    mg_mce_after "$b" "$k"
    [ -e "$SYS/bus/pci/devices/$b" ] && { back=$((back + 1)); continue; }
    if [ -n "$rp" ] && ! rk_gate; then mg_say "recovery of $(mg_tag "$b") skipped: $RK_GATE_WHY"; continue; fi
    [ -n "$rp" ] && echo 1 | mg_rto 20 tee "$SYS/bus/pci/devices/$rp/rescan" > /dev/null 2>&1
    if [ ! -e "$SYS/bus/pci/devices/$b" ] && [ -n "$rp" ] && [ "${MG_SBR:-1}" = 1 ] &&
       [ "$(ls -d "$SYS/bus/pci/devices/$rp"/0000:* 2>/dev/null | grep -vc "/$b\$")" = 0 ]; then
      mg_sbr "$rp" && mg_link_wait "$rp"
      echo 1 | mg_rto 20 tee "$SYS/bus/pci/devices/$rp/rescan" > /dev/null 2>&1
    fi
    if [ -e "$SYS/bus/pci/devices/$b" ]; then back=$((back + 1)); mg_say "recovery: $(mg_tag "$b") back on PCI after its root port ${rp:-?} rescan"
    else mg_say "recovery: $(mg_tag "$b") still off PCI after its root port ${rp:-?} rescan; the reset path (glx_reset, tray re-power) follows"; fi
  done
  declare -F rk_remap > /dev/null && rk_remap
  mg_say "recovery: $back of $n removed chip(s) back on PCI before the reset; $RK_N chips on the map"
  # the heavy-logs captures held back by mg_defer_heavy: BMC/SEL/SDR/lspci/dmesg, after the rescan; no device open
  # (HEAVY_DEVICE_READS=0) and no hwmon read of a chip still hung (MG_HUNG): the reset has not run yet
  if [ ${#MG_HEAVY_LATER[@]} -gt 0 ] && declare -F heavy_logs_snapshot > /dev/null; then
    for e in "${MG_REMOVED[@]}"; do b=${e%%|*}; [[ " $MG_HUNG " == *" $b "* ]] || MG_HUNG="${MG_HUNG:+$MG_HUNG }$b"; done
    HEAVY_DEVICE_READS=0 heavy_logs_snapshot hang_after_rescan > /dev/null 2>&1
    mg_say "heavy-logs hang capture ran after the rescan (${MG_HEAVY_LATER[*]}): no device open, no hwmon read of the hung chip(s)"
    MG_HEAVY_LATER=()
  fi
  MG_RECOVERED=$n; MG_REMOVED=(); MG_MCE_WARNED=; MG_T0=$(date +%s)
}
mg_try_chip_hang() {  # <try n> <try log>: a plain ub0_repro try that ended on its own CHIP_HANG (exit 3, not past its
                      # wall-clock limit): the guard on the chips its RESULT line names, timed from its FIRST LOSS line
  local l f t0 bdfs
  l=$(grep -a 'loss=CHIP_HANG' "$2" 2> /dev/null | tail -1); [ -n "$l" ] || return 1
  bdfs=$(grep -oE 'pci=\[[^]]*\]' <<< "$l" | grep -oE '[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]' | sort -u)
  f=$(grep -a -m1 -oE 'FIRST LOSS at [0-9T:.-]+' "$2" | sed 's/^FIRST LOSS at //')
  t0=$([ -n "$f" ] && date -u -d "$f" +%s 2> /dev/null); [ -n "$t0" ] || t0=$(date +%s)
  # shellcheck disable=SC2086  # a word list of BDFs
  MG_TRYLOSS=1 mg_hang "try$1" "$t0" "" $bdfs
}
mg_report() {  # the summary line
  local n; mg_release; n=$(grep -c $'\tremoved\t' "$(mg_dir)/timeline.tsv" 2>/dev/null); n=${n:-0}
  if [ "$n" -gt 0 ]; then say "MCE guard: removed $n hung chip(s) from PCI: $(awk -F'\t' '$2 == "removed" { print $4 " in " $1 " at " $3 " s" }' "$(mg_dir)/timeline.tsv" | paste -sd';' | sed 's/;/; /g') ($(mg_dir)/timeline.tsv)"
  else say "MCE guard: $MG_WHY; no chip removed"; fi
}
