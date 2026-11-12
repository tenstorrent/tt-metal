# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/sudo_route.sh: the no-sudo warning that run.sh and reliability.sh print as their first output line.
# sudo_route_check <part>...: asks sudo only `sudo -n -l <exact command>` (it lists rules and runs
# nothing; the rescan part also runs a bare `sudo -n -l`), each under a 5 s cap with stdin closed: it never prompts, never blocks and never runs a privileged command.
# Parts: hsc (under-voltage/HSC reads: VIN, IOUT, STATUS_*), clear (the HSC CLEAR_FAULTS write), cpld (fleet CPLD/AER reads),
# sensors (BMC sensor reads), drop (drop-time CPLD read and pre-reset HSC/CPLD dump), rescan (root-port rescan after
# glx_reset), cycles (power cycles), recovery (recovery cycles). The probe cap is 5 s (SR_PROBE_S, for tests only).
# Sets SR_STATE (full, partial or none), SR_LINE (the WARN line; empty with full sudo) and SR_TAG (the RESULT line suffix;
# empty with full sudo). Each part is judged by the check its own reader uses: every reader probes its exact command
# with `sudo -n -l` (never `sudo -n true`, which a fixed-argument sudoers list refuses), so a
# part works when sudo lists its command. Known gap: -l also lists a rule that wants a password. The rescan part alone
# is judged as run.sh's rk_can_root does: root, or NOPASSWD: ALL in `sudo -n -l`.
SR_STATE="" SR_LINE="" SR_TAG=""
SR_PROBE_S=${SR_PROBE_S:-5}
sr_label() { case $1 in
  hsc) echo "under-voltage/HSC (VIN, IOUT) reads" ;; clear) echo "HSC CLEAR_FAULTS" ;; cpld) echo "CPLD reads" ;; drop) echo "drop-time CPLD read and pre-reset HSC/CPLD dump" ;;
  sensors) echo "BMC sensor reads" ;; rescan) echo "root-port rescan after glx_reset" ;;
  cycles) echo "power cycles" ;; recovery) echo "recovery cycles" ;; *) echo "$1" ;; esac; }
sr_cmds() { case $1 in   # one exact command per line; the part works when any one is listed
  hsc) echo "ipmitool raw 0x06 0x52 0x11 0x40 0x02 0x79" ;;
  clear) echo "ipmitool raw 0x06 0x52 0x11 0x40 0x00 0x03" ;;
  cpld) echo "ipmitool raw 0x06 0x52 0x11 0x80 0x01 0x1E" ;;
  sensors) echo "ipmitool sdr type Power Supply" ;;
  drop) echo "ipmitool raw 0x06 0x52 0x11 0x80 0x01 0x1E"; echo "ipmitool raw 0x06 0x52 0x11 0x40 0x02 0x79" ;;
  rescan) echo "tee /sys/bus/pci/rescan" ;;
  cycles|recovery) echo "ipmitool raw 0x06 0x52 0x09 0x62 0x00 0x14 0x5F 0x54 0x37 0x53"; echo "ipmitool chassis power cycle" ;;
  esac; }
sr_join() { local o="" x; for x in "$@"; do o+="${o:+, }$x"; done; echo "$o"; }
sudo_route_check() {
  local host p c w ok=() off=() l a; host=$(hostname -s 2> /dev/null || echo "?")
  [ "$(id -u)" = 0 ] && { SR_STATE=full SR_LINE="" SR_TAG=""; return 0; }
  for p in "$@"; do
    l=no
    if [ "$p" = rescan ]; then  # same rule as run.sh rk_can_root: a tee-only rule does not count
      command -v sudo > /dev/null 2>&1 && timeout "$SR_PROBE_S" sudo -n -l < /dev/null 2> /dev/null | grep -qE 'NOPASSWD: *(.*, *)?ALL *(,|$)' && l=yes
    elif command -v sudo > /dev/null 2>&1; then
      while IFS= read -r c; do
        [ -n "$c" ] || continue; read -ra a <<< "$c"; [ "$c" = "ipmitool sdr type Power Supply" ] && a=(ipmitool sdr type "Power Supply")
        timeout "$SR_PROBE_S" sudo -n -l "${a[@]}" < /dev/null > /dev/null 2>&1 && { l=yes; break; }
      done < <(sr_cmds "$p")
    fi
    if [ "$l" = yes ]; then ok+=("$(sr_label "$p")"); else off+=("$(sr_label "$p")"); fi
  done
  if [ ${#off[@]} = 0 ]; then SR_STATE=full SR_LINE="" SR_TAG=""
  elif [ ${#ok[@]} = 0 ]; then SR_STATE=none
    SR_LINE="WARNING: NO SUDO on $host: $(sr_join "${off[@]}") are DISABLED for this run"
    SR_TAG="; NO SUDO: $(sr_join "${off[@]}") disabled"
  else SR_STATE=partial
    SR_LINE="WARNING: PARTIAL SUDO on $host: works: $(sr_join "${ok[@]}"); DISABLED for this run: $(sr_join "${off[@]}")"
    SR_TAG="; PARTIAL SUDO: $(sr_join "${off[@]}") disabled"
  fi
  return 0; }
