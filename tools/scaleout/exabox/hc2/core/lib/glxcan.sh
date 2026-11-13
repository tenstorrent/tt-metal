# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/glxcan.sh: rk_sudo_l, rk_can_root, rk_can_glx for lib/fleet.sh run without run.sh (tests). Same text as run.sh's;
# fleet.sh sources this only where run.sh has not defined them. Keep both in step.
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
