# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/kmd_policy.sh: the tenstorrent KMD's power_policy, sourced by run.sh.
# kp_read: the value in every run's header (read only). --kmd-power-policy=<v> (opt-in, every mode the same way): before
# C2, kp_apply first runs this host's own busy gate (kp_busy: run.sh's run lock, its P0 gate and reset gate verdicts, the
# user's hold file, then fleet/gate.sh; this host only), then checks that sudo -n allows modprobe and that
# no other process holds the chips (any refusal: exit 11, no modprobe, nothing changed, kp_restore reloads nothing), reads every chip's power1_max/power1_cap, reloads the module with power_policy=<v>, recounts the chips by BDF
# and tray (a chip missing then is left to the start reset, glx_reset with its root-port rescan, goal 9) and re-reads the
# limits: any limit that moved is a FINDING by BDF and tray, the default driver comes back and the run ends (exit 11).
# kp_restore (run.sh's EXIT trap, after any exit: success, failure, timeout's TERM, Ctrl-C) reloads the module with no
# parameter and logs the power_policy it then reads. It never writes a power limit, and never kills or signals a holder.
# Test hooks: TRAY_KMD_PARAM (the power_policy file), KP_PROC (/proc), KP_DEVD (/dev/tenstorrent), KP_REFCNT (the module's
# refcnt file), KP_SETTLE_S (seconds to wait for the chips' tt-kmd nodes after the reload, default 30).
KP_WANT=""; KP_ARMED=0; KP_BASE=""
KP_GATE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../fleet/gate.sh   # this host's own busy gate (fleet runs' P0 gate)
kp_say() { if declare -F say > /dev/null; then say "$@"; else echo "$*"; fi; }
kp_param() { echo "${TRAY_KMD_PARAM:-${SYS:-/sys}/module/tenstorrent/parameters/power_policy}"; }
kp_read() {  # the loaded KMD's power_policy, or why there is none
  local f v; f=$(kp_param)
  if [ -r "$f" ] && read -r v < "$f" 2> /dev/null; then echo "$v"; else echo "unknown (no $f)"; fi
}
kp_tray() { if declare -F rk_tray > /dev/null; then rk_tray "$1"; else echo "?"; fi; }
kp_bdfs() {  # every Blackhole function on PCIe (vendor 0x1e52)
  local d v; for d in "${SYS:-/sys}"/bus/pci/devices/*; do v=; read -r v < "$d/vendor" 2> /dev/null
    [ "$v" = 0x1e52 ] && echo "${d##*/}"; done; return 0
}
kp_limits() {  # "<bdf> <power1_max> <power1_cap>" per chip on PCIe ("-" where its hwmon has none); read only
  local b h m c
  for b in $(kp_bdfs); do h=$(ls -d "${SYS:-/sys}/bus/pci/devices/$b"/hwmon/hwmon* 2> /dev/null | head -1)
    m=-; c=-; [ -n "$h" ] && { read -r m < "$h/power1_max" 2> /dev/null || m=-; read -r c < "$h/power1_cap" 2> /dev/null || c=-; }
    echo "$b $m $c"; done
}
kp_ours() {  # $1 a pid: 0 when this run ($$) is among its ancestors
  local p=$1 s l; while [ -n "$p" ] && [ "$p" -gt 1 ] 2> /dev/null; do [ "$p" = "$$" ] && return 0
    read -r l 2> /dev/null < "${KP_PROC:-/proc}/$p/stat" || return 1; s=${l##*) }; s=${s#* }; p=${s%% *}; done; return 1
}
kp_holders() {  # "pid (comm, user)" of every other process with a /dev/tenstorrent node open; read only (no fuser/lsof)
  local pr=${KP_PROC:-/proc} dd=${KP_DEVD:-/dev/tenstorrent} f p seen=" " fnd=(find)
  [ "$(id -u)" != 0 ] && sudo -n -l find > /dev/null 2>&1 && fnd=(sudo -n find)   # other users' fds need root
  while IFS= read -r f; do p=${f#"$pr"/}; p=${p%%/*}; [[ $seen == *" $p "* ]] && continue; seen+="$p "
    kp_ours "$p" && continue
    echo "$p ($(cat "$pr/$p/comm" 2> /dev/null || echo ?), $(stat -c %U "$pr/$p" 2> /dev/null || echo ?))"
  done < <(timeout 30 "${fnd[@]}" "$pr"/[0-9]*/fd -maxdepth 1 \( -lname "$dd/*" -o -lname "$dd" \) 2> /dev/null)
  return 0
}
kp_refuse() {  # nothing changed: a plain line, the verdict, exit 11
  kp_say "--kmd-power-policy=$KP_WANT refused: $*; nothing changed, power_policy stays $(kp_read)"
  declare -F vd_say > /dev/null && vd_say "NOT RUN (--kmd-power-policy refused: $*)"; exit 11
}
kp_busy() {  # 0 and the reason when this host is busy or held: never unload the module under live H3 or another run
  local o pg   # tdp_refused's conditions first, then the hold file, then fleet/gate.sh
  if [ "${TRAY_RUN_LOCK_BUSY:-0}" = 1 ]; then echo "run lock held by another run"
  elif [[ ${FL_STOP:-} == "gate busy"* ]]; then echo "$FL_STOP"
  elif [ -n "${RK_GATE_WHY:-}" ]; then echo "$RK_GATE_WHY"
  elif declare -F hf_check > /dev/null && ! hf_check 2> /dev/null; then echo "hold: ${HF_WHY:-}"
  else pg=$(ps -o pgid= -p $$ 2> /dev/null | tr -d ' ')
    if [ ! -r "$KP_GATE" ]; then echo "no own-host gate ($KP_GATE missing)"
    elif ! o=$(FLEET_PROC=${FLEET_PROC:-/proc} bash "$KP_GATE" "$pg" 2>&1); then o=$(grep -a 'gate busy' <<< "$o" | tail -1); echo "${o:-own-host gate failed}"
    else return 1; fi
  fi
}
kp_apply() {  # $1 the power_policy to load (0 or 1); the steps before C2
  local v=$1 old held refc n t miss b m c bm bc moved="" gone=""
  KP_WANT=$v; old=$(kp_read)
  local why; if why=$(kp_busy); then kp_refuse "this host is busy or held ($why); the module is never unloaded under live H3 or another run"; fi
  kp_say "--kmd-power-policy=$v: power_policy is $old; reloading the tenstorrent module with power_policy=$v before C2"
  sudo -n -l modprobe > /dev/null 2>&1 || kp_refuse "sudo -n does not allow modprobe here (sudo -n -l modprobe)"
  held=$(kp_holders)
  refc=$(cat "${KP_REFCNT:-${SYS:-/sys}/module/tenstorrent/refcnt}" 2> /dev/null || echo "?")
  [ -n "$held" ] && kp_refuse "other processes hold /dev/tenstorrent: $(echo "$held" | paste -sd, | sed 's/,/, /g') (never killed or signalled)"
  [ "$refc" = 0 ] || [ "$refc" = "?" ] || kp_refuse "the tenstorrent module is in use (refcnt $refc) by a holder this scan cannot see"
  KP_BASE=$(kp_limits)
  kp_say "power limits before the reload (bdf power1_max power1_cap): $(echo "$KP_BASE" | paste -sd, | sed 's/,/, /g')"
  sudo -n modprobe -r tenstorrent || kp_refuse "sudo -n modprobe -r tenstorrent failed (rc $?; the module stays loaded)"
  KP_ARMED=1   # unloaded: from here every exit puts the default driver back (kp_restore, the EXIT trap)
  kp_say "change: tenstorrent module reloaded with power_policy=$v (was $old); undo: sudo -n modprobe -r tenstorrent && sudo -n modprobe tenstorrent (runs at this run's end, also on a failure, a timeout or Ctrl-C)"
  if ! sudo -n modprobe tenstorrent "power_policy=$v"; then
    kp_say "FINDING KMD RELOAD: sudo -n modprobe tenstorrent power_policy=$v failed; restoring the default driver"
    kp_restore; declare -F vd_say > /dev/null && vd_say "NOT RUN (--kmd-power-policy: the module would not load with power_policy=$v)"; exit 11
  fi
  t=$(( $(date +%s) + ${KP_SETTLE_S:-30} ))   # each chip's tt-kmd node, as rk_state counts it
  while :; do miss=""; n=0
    for b in $(echo "$KP_BASE" | awk '{print $1}'); do
      if compgen -G "${SYS:-/sys}/bus/pci/devices/$b/tenstorrent/tenstorrent!*" > /dev/null; then n=$((n + 1)); else miss+="${miss:+, }$b (tray $(kp_tray "$b"))"; fi
    done; [ -z "$miss" ] || [ "$(date +%s)" -ge "$t" ] && break; sleep 1; done
  kp_say "after the reload: power_policy $(kp_read); $n/$(echo "$KP_BASE" | grep -c .) chips have a tt-kmd node${miss:+; missing: $miss (left to the start reset: glx_reset with its root-port rescan)}"
  while read -r b m c; do [ -n "$b" ] || continue
    if [[ $miss == *"$b "* ]]; then gone+="${gone:+, }$b"; continue; fi
    read -r bm bc < <(echo "$KP_BASE" | awk -v b="$b" '$1 == b {print $2, $3}')
    [ "$m" = "$bm" ] && [ "$c" = "$bc" ] || moved+="${moved:+$'\n'}FINDING KMD POWER LIMIT $b (tray $(kp_tray "$b")): power1_max $bm -> $m, power1_cap $bc -> $c after the power_policy=$v reload"
  done < <(kp_limits)
  [ -n "$gone" ] && kp_say "limits not re-read (no tt-kmd node yet): $gone"
  if [ -n "$moved" ]; then
    while IFS= read -r l; do kp_say "$l"; done <<< "$moved"
    kp_say "power limits moved after the reload: restoring the default driver and ending the run (no limit is ever written)"
    kp_restore; declare -F vd_say > /dev/null && vd_say "FAIL (KMD power limit moved after the power_policy=$v reload)"; exit 11
  fi
  kp_say "power limits unchanged after the reload ($(( n )) chips re-read)"
  return 0
}
kp_restore() {  # the default driver back (idempotent; the EXIT trap and the aborts above)
  [ "${KP_ARMED:-0}" = 1 ] || return 0; KP_ARMED=0
  local rc=0 why=""
  if ! sudo -n modprobe -r tenstorrent && [ -e "$(kp_param)" ]; then rc=1; why="modprobe -r failed, the power_policy=$KP_WANT module is still loaded"
  elif ! sudo -n modprobe tenstorrent; then rc=1; why="modprobe tenstorrent failed"; fi
  if [ "$rc" = 0 ]; then kp_say "KMD restore ok: tenstorrent module reloaded with no parameter; power_policy now $(kp_read)"
  else kp_say "KMD RESTORE FAILED ($why): power_policy now $(kp_read); undo by hand: sudo modprobe -r tenstorrent && sudo modprobe tenstorrent"; fi
  return 0
}
