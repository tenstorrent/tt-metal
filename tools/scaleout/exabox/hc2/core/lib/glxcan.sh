# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/glxcan.sh: rk_sudo_l, rk_can_root, rk_can_glx for lib/fleet.sh run without run.sh (tests). Same text as run.sh's;
# fleet.sh sources this only where run.sh has not defined them. Keep both in step. rk_settle_open (below) is only here:
# reliability.sh sources this file for it after every glx_reset.
rk_sudo_l() { [ -n "${RK_SUDO_L+x}" ] || RK_SUDO_L=$(timeout 15 sudo -n -l 2> /dev/null); printf '%s\n' "$RK_SUDO_L"; }
# rk_can_root: root, or sudo -n -l lists ALL (sysfs rescan, tt-kmd re-probe, any ipmitool command)
rk_can_root() { [ "$(id -u)" = 0 ] || rk_sudo_l | grep -qE 'NOPASSWD: *(.*, *)?ALL *(,|$)'; }
# rk_glx_host_ok: the explicit allow list for a glx_reset that runs on sudoers rules short of ALL: the host names the
# pattern below matches. Anything else keeps NO RESET.
declare -F hc_site > /dev/null || . "$(dirname "${BASH_SOURCE[0]}")/hc.sh"   # the allow list is site data (glx_hosts)
rk_glx_host_ok() { local h=${TRAY_REPRO_HOST:-$(hostname -s)} re; re=$(hc_site glx_hosts)
  [ -n "$re" ] && [[ $h =~ ^($re)$ ]]; }
# rk_can_glx: how tt-smi -glx_reset can run here. RK_GLX_SUDO: "" (tt-smi as is; root, sudo ALL, or sudo -n -l lists
# the ipmitool raw 0x30 0x8B forms tt-smi issues, A67/A89-style) or "sudo -n" (sudo -n -l lists tt-smi).
# 1 with RK_GLX_CAN_WHY when none applies: then no reset runs at all.
rk_can_glx() {
  local smi=${TRAY_REPRO_TT_SMI:-tt-smi} l p; RK_GLX_SUDO=""; RK_GLX_RAWONLY=0; RK_GLX_CAN_WHY=""
  command -v "$smi" > /dev/null || { RK_GLX_CAN_WHY="$smi not found"; return 1; }
  rk_can_root && return 0
  rk_glx_host_ok || { RK_GLX_CAN_WHY="no root, and this host is not on the glx_reset allow list (sudo rules alone do not decide)"; return 1; }
  l=$(rk_sudo_l); p=$(command -v "$smi")
  if grep -qE "NOPASSWD:.*(${p//./\\.}|/tt-smi)( |,|$)" <<< "$l"; then RK_GLX_SUDO="sudo -n"; return 0; fi
  if grep -qiE 'NOPASSWD:.*ipmitool raw 0x30 0x8b' <<< "$l"; then RK_GLX_RAWONLY=1; return 0; fi
  RK_GLX_CAN_WHY="no root, and sudo -n -l lists neither tt-smi nor ipmitool raw 0x30 0x8B"; return 1
}

# rk_settle_open <what> <chips wanted> <settle s> [<answer cap s> [<sysfs root>]]: after a glx_reset (and its root-port
# rescan), before any chip is opened (tt-umd discovery, an open probe, a load): wait until every wanted chip answers on PCIe
# (config word 1e52 under <sysfs root>/bus/pci/devices, polled every 2 s, at most the cap, 120 s by default), then
# wait the settle seconds more (the CMFW reinit of a chip opened 1-2 s after the reset failed 9 of 100 times in the
# 11-16 A/B loop, #12672; #12711's rule is all chips answer, then 60 s). Sets one line that names both waits. A chip that
# never answers does not stop the settle: the caller's recount names it. Sets RK_SETTLE_LINE (the caller logs it; no
# subshell, so a caller's trap still runs during the waits). Always returns 0.
rk_settle_open() {
  local what=$1 want=$2 settle=$3 cap=${4:-120} sys=${5:-/sys} t0 n d w
  t0=$(date +%s)
  while :; do
    n=0; for d in "$sys"/bus/pci/devices/*; do
      w=$(timeout 5 od -An -tx2 -N2 "$d/config" 2> /dev/null | tr -d ' \n' || true); [ "$w" = 1e52 ] && n=$((n + 1)); done
    [ "$n" -ge "$want" ] && break
    [ $(( $(date +%s) - t0 )) -ge "$cap" ] && break
    sleep 2 & wait $! || true
  done
  RK_SETTLE_LINE="settle after $what: $n/$want chips answer after $(( $(date +%s) - t0 )) s; waiting $settle s more before any chip is opened"
  [ "$settle" -gt 0 ] && { sleep "$settle" & wait $! || true; }
  return 0
}
