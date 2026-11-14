#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/tdp.sh: run.sh --tdp=<W> (no refusal, no range check, the operator sets it).
# tt-metal (since 2026-08-20) reads TT_METAL_TDP_LIMIT_WATTS at every cluster open and sets each PCIe chip's
# firmware TDP limit to it (firmware 19.11.0 or newer; 0 = the board default). Nothing reverts it on close; a board
# reset does. This file only passes the variable on and reads each chip's limit back from sysfs (hwmon power1_max, uW);
# it never writes a power limit itself. Each chip's configured chip_limits.tdp_limit (the per-rack budget) and
# max_tdp_limit (the hard cap tt-metal's request is capped at) are only read (lib/tdp_cfg.py).
# Sourced by run.sh (uses TDP, SYS, OUT, PY, AS, say, rk_tray):
#   tdp_start    right after the own-host busy gate passed (fl_start's P0 gate; --mode=repro: after rk_start), so this
#                run holds the chips: restore a limit a crashed run left (its marker), then with --tdp write the marker,
#                export TT_METAL_TDP_LIMIT_WATTS (TDP_SET=1), print the setup lines and any no-effect WARN. Without --tdp
#                and without a marker it does nothing. Gate busy or held (tdp_refused): nothing opens, the marker stays.
#   tdp_applied  before the end reset: each chip's limit read back by BDF and tray 1-4 (the limit the load ran under)
#   tdp_end      after the end reset, before the RESULT block: restore if any chip is off its start limit, print the summary
#   tdp_exit     the EXIT trap: on an error, INT or TERM before tdp_end, restore and print the restored value
#   tdp_tag      the RESULT line's '; TDP: ran at the default TDP ..., not --tdp=<W>: gate refused (...)' after a refused tdp_start
# Only a run that set the limit (TDP_SET=1) restores it, and never after its gate said busy or held (a --tdp run refused as busy opened the chips another run held for a restore it never needed).
# Env: TRAY_TDP_MARKER (default <run.sh dir>/.tdp_applied.<host>), TRAY_TDP_RESTORE_S (60), TRAY_TDP_TTNN_DIR (the ttnn
# package dir to look in for TT_METAL_TDP_LIMIT_WATTS support; default: found from $PY without importing it).
TDP=${TDP:-}; TDP_ON=0; TDP_SET=0; TDP_TRIED=0; TDP_DONE=0; TDP_START=""; TDP_APPLIED=""; TDP_CFG=""; TDP_REFUSED=""; TDP_REF_MARK=0; TDP_REF_READ=""
TDP_MARKER=${TRAY_TDP_MARKER:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.tdp_applied.$(hostname -s 2>/dev/null || echo host)}
tdp_say() { if declare -F say > /dev/null; then say "$@"; else echo "$*"; fi; }
tdp_tray() {  # rk_tray where run.sh defines it; alone: the bus byte by rk_bus's regex, '?' for an address without one
  if declare -F rk_tray > /dev/null; then rk_tray "$1"
  elif [[ $1 =~ ^([0-9a-fA-F]{4,}:)?([0-9a-fA-F]{2}):[0-9a-fA-F]{2}\.[0-7]$ ]]; then
    case $(( 16#${BASH_REMATCH[2]} & 0xF0 )) in 0) echo 1 ;; 64) echo 2 ;; 192) echo 3 ;; 128) echo 4 ;; *) echo "?" ;; esac
  else echo "?"; fi; }
tdp_read() {  # per chip: its BDF and watts (- if unread), from hwmon power1_max (uW); read-only
  local d h b v
  for d in "${SYS:-/sys}"/class/tenstorrent/tenstorrent!*; do
    [ -e "$d" ] || continue
    b=$(basename "$(readlink -f "$d/device")"); b=${b#0000:}
    h=$(ls -d "$d"/device/hwmon/hwmon* 2>/dev/null | head -1)
    v=$(cat "$h/power1_max" 2>/dev/null) && [[ $v =~ ^[0-9]+$ ]] && v=$(( v / 1000000 )) || v=-
    echo "$b $v"
  done | sort
}
tdp_fmt() {  # <"bdf W" lines>: "160 W on 32 chips" or "160 W on 31 chips; 450 W on 02:00.0 (tray 1)"
  local l b w; declare -A n=() who=()
  while read -r b w; do [ -n "$b" ] || continue; n[$w]=$(( ${n[$w]:-0} + 1 )); who[$w]="${who[$w]:+${who[$w]}, }$b (tray $(tdp_tray "$b"))"; done <<< "$1"
  [ ${#n[@]} -gt 0 ] || { echo "no chip read"; return; }
  for w in $(for w in "${!n[@]}"; do echo "${n[$w]} $w"; done | sort -rn | awk '{ print $2 }'); do
    l="${l:+$l; }$w$([ "$w" = - ] && echo " (unread)" || echo " W") on"
    if [ ${#n[@]} = 1 ] || [ "${n[$w]}" -gt 8 ]; then l="$l ${n[$w]} chips"; else l="$l ${who[$w]}"; fi
  done; echo "$l"
}
tdp_cfg() {  # "<bdf> <tdp_limit> <max_tdp_limit>" per chip (lib/tdp_cfg.py, a pyluwen SPI table read); nothing without pyluwen
  local p py=${TRAY_TDP_CFG_PY:-}
  if [ -z "$py" ]; then for p in /opt/tenstorrent/tt-tools/tt-flash/.venv/bin/python "${PY:-python3}"; do
    "$p" -c 'import pyluwen' > /dev/null 2>&1 && { py=$p; break; }; done; fi
  [ -n "$py" ] || return 0
  timeout "${TRAY_TDP_CFG_S:-60}" ${AS[@]+"${AS[@]}"} "$py" "$(dirname "${BASH_SOURCE[0]}")/tdp_cfg.py" 2> /dev/null | grep -E '^[0-9a-f]+:[0-9a-f]+\.[0-9] ' | sort || true
}
tdp_chips() {  # <"bdf W" read-back lines> <label>: per chip by BDF and tray 1-4, configured limits, the request, the read-back
  local b w t m
  while read -r b w; do [ -n "$b" ] || continue
    read -r t m <<< "$(awk -v b="$b" '$1 == b { print $2, $3 }' <<< "$TDP_CFG")"
    tdp_say "TDP chip $b (tray $(tdp_tray "$b")): tdp_limit ${t:--} W, max_tdp_limit ${m:--} W, requested $TDP W, $2 $w W"
  done <<< "$1"
}
tdp_refused() {  # 0 and the reason when this run's own-host gate said busy or held: no TDP open may run
  if [ "${TRAY_RUN_LOCK_BUSY:-0}" = 1 ]; then echo "run lock held by another run"
  elif [[ ${FL_STOP:-} == "gate busy"* ]]; then echo "${FL_STOP}"
  elif [ -n "${RK_GATE_WHY:-}" ]; then echo "$RK_GATE_WHY"
  else return 1; fi
}
tdp_restore() {  # <why> [keep]: one short open with TT_METAL_TDP_LIMIT_WATTS=0 (each chip back to its board default), then read back
  local rc=0 now
  tdp_say "TDP restore ($1): one short tt-metal open with TT_METAL_TDP_LIMIT_WATTS=0 (board default)"
  TT_METAL_TDP_LIMIT_WATTS=0 timeout "${TRAY_TDP_RESTORE_S:-60}" ${AS[@]+"${AS[@]}"} ${AS:+TT_METAL_TDP_LIMIT_WATTS=0} "${PY:-python3}" -c \
    'import ttnn; print("TDP restore: devices", ttnn.GetNumAvailableDevices())' > "${OUT:-/tmp}/tdp_restore.log" 2>&1 || rc=$?
  now=$(tdp_read)
  if [ "$rc" = 0 ]; then tdp_say "TDP restored: $(tdp_fmt "$now")${TDP_START:+ (start: $(tdp_fmt "$TDP_START"))}"; rm -f "$TDP_MARKER"
  else tdp_say "TDP WARN: the restore open failed (rc $rc, ${OUT:-/tmp}/tdp_restore.log); limit now $(tdp_fmt "$now"); a tt-smi -glx_reset restores the board default$([ "${2:-}" = keep ] && echo "; marker $TDP_MARKER kept for the next run")"; fi
  return 0
}
tdp_start() {
  local m fw lib b why
  TDP_TRIED=1
  if why=$(tdp_refused); then   # never reached on a P0 refusal (exit 10 first); a guard for any other caller
    if [ -n "$TDP" ]; then TDP_REFUSED=$why; TDP_REF_READ=$(tdp_read); [ -f "$TDP_MARKER" ] && TDP_REF_MARK=1; fi   # for tdp_tag: a later gate resets RK_GATE_WHY
    [ -n "$TDP" ] || [ -f "$TDP_MARKER" ] && tdp_say "TDP: gate refused ($why): no TDP open${TDP:+, --tdp=$TDP not applied}$([ -f "$TDP_MARKER" ] && echo "; marker $TDP_MARKER kept for the next run")"
    return 0; fi
  if [ -f "$TDP_MARKER" ]; then
    m=$(head -1 "$TDP_MARKER" 2>/dev/null)
    tdp_say "TDP: a previous run left --tdp applied ($m; marker $TDP_MARKER): restoring the default before this run"
    tdp_restore "marker of a run that did not finish"; rm -f "$TDP_MARKER"   # this run's own resets restore it too
  fi
  [ -n "$TDP" ] || return 0
  TDP_ON=1; TDP_START=$(tdp_read)
  echo "--tdp=$TDP $(date '+%F %T %Z') pid $$ ${OUT:-}" > "$TDP_MARKER" 2>/dev/null || tdp_say "TDP WARN: cannot write the marker $TDP_MARKER"
  export TT_METAL_TDP_LIMIT_WATTS=$TDP
  [ -n "${AS[0]+x}" ] && AS+=(TT_METAL_TDP_LIMIT_WATTS="$TDP")   # sudo -u ... env drops the caller's environment
  [ -n "${AS[0]+x}" ] && [ -n "${HEAVY_AS+x}" ] && HEAVY_AS="${AS[*]}"   # set before this call (run.sh)
  TDP_SET=1   # this run applied the limit: only now may tdp_end or tdp_exit restore it
  tdp_say "TDP: --tdp=$TDP: TT_METAL_TDP_LIMIT_WATTS=$TDP for every tt-metal process of this run; limit at start: $(tdp_fmt "$TDP_START"); restored at the end"
  TDP_CFG=$(tdp_cfg)
  [ -n "$TDP_CFG" ] || tdp_say "TDP: tdp_limit and max_tdp_limit not read (no pyluwen route for lib/tdp_cfg.py)"
  tdp_chips "$TDP_START" "read back at start"
  m=$(awk -v w="$TDP" '$3 ~ /^[0-9]+$/ && $3 + 0 < w + 0 { print $1, $3 }' <<< "$TDP_CFG")
  if [ -n "$m" ]; then
    [ "$(wc -l <<< "$m")" -gt 8 ] && m=$(tdp_fmt "$m") || m=$(while read -r b fw; do echo "$b (tray $(tdp_tray "$b")) $fw W"; done <<< "$m" | paste -sd, | sed 's/,/, /g')
    tdp_say "TDP WARN: --tdp=$TDP is above max_tdp_limit on $m: tt-metal caps the limit at max_tdp_limit"; fi
  lib=${TRAY_TDP_TTNN_DIR:-$("${PY:-python3}" -c 'import importlib.util as u, os; s = u.find_spec("ttnn"); print(os.path.dirname(s.origin) if s and s.origin else "")' 2>/dev/null)}
  if [ -z "$lib" ] || ! grep -raqs --include='*.so*' TT_METAL_TDP_LIMIT_WATTS "$lib"; then
    tdp_say "TDP WARN: --tdp had no effect: this tt-metal (${lib:-ttnn not found}) is too old to take a power limit (no TT_METAL_TDP_LIMIT_WATTS)"
  fi
  fw=$( [ -s "${OUT:-}/fw.txt" ] && declare -F fw_major > /dev/null && fw_major "$OUT/fw.txt")
  if [ -n "$fw" ] && [ "$fw" != - ] && [ "$(printf '%s\n' "$fw" 19.11.0 | sort -V | head -1)" != 19.11.0 ]; then
    tdp_say "TDP WARN: --tdp had no effect: firmware $fw is older than 19.11.0"
  fi
  return 0
}
tdp_applied() { [ "$TDP_SET" = 1 ] || return 0; TDP_APPLIED=$(tdp_read); return 0; }
tdp_skip() {  # <when>: a run that set no limit, or whose gate said busy or held, opens nothing; one line
  local why
  if [ "$TDP_SET" = 1 ] && why=$(tdp_refused); then
    tdp_say "TDP: restore skipped ($1): gate refused ($why); marker $TDP_MARKER kept for the next run; a tt-smi -glx_reset restores the board default"
  elif [ "$TDP_SET" != 1 ]; then
    [ -n "$TDP" ] && tdp_say "TDP: not applied by this run, restore skipped"
    [ "$TDP_TRIED" != 1 ] && [ -f "$TDP_MARKER" ] && tdp_say "TDP: marker $TDP_MARKER of an earlier run kept: this run's gate or start check did not pass, no TDP open ran"
  fi; return 0
}
tdp_tag() {  # '; TDP: ...' for the RESULT line when --tdp was given but tdp_start's gate refused it:
             # the load ran at the limit the box already had, read back then, so the run is never read as tested at --tdp
  [ -n "$TDP" ] && [ -n "$TDP_REFUSED" ] && [ "$TDP_SET" != 1 ] || return 0
  if [ "$TDP_REF_MARK" = 1 ]; then echo "; TDP: ran at the limit the box had (an earlier run's --tdp marker kept; read back at start: $(tdp_fmt "$TDP_REF_READ")), not --tdp=$TDP: gate refused ($TDP_REFUSED)"
  else echo "; TDP: ran at the default TDP (read back at start: $(tdp_fmt "$TDP_REF_READ")), not --tdp=$TDP: gate refused ($TDP_REFUSED)"; fi
}
tdp_end() {
  [ "$TDP_DONE" = 0 ] || return 0
  if [ "$TDP_SET" != 1 ] || tdp_refused > /dev/null; then TDP_DONE=1; tdp_skip "end of run"; return 0; fi
  TDP_DONE=1
  local now fwl
  fwl=$(timeout 30 grep -rahoE 'firmware TDP limit on chip [0-9]+ is now [0-9]+ W' "${OUT:-/nonexistent}" 2>/dev/null | sort -u | awk '{ print $(NF-1) }' | sort -n | uniq -c | awk '{ printf "%s%s W on %d chip(s)", (NR > 1 ? "; " : ""), $2, $1 }')
  tdp_say "TDP: requested --tdp=$TDP; read back during the run: $(tdp_fmt "${TDP_APPLIED:-$(tdp_read)}"); tt-metal log: ${fwl:-no 'firmware TDP limit' line}"
  tdp_chips "${TDP_APPLIED:-$(tdp_read)}" "read back during the run"
  now=$(tdp_read)
  if [ -z "$TDP_START" ] || [ "$now" != "$TDP_START" ]; then tdp_restore "end of run"
  else tdp_say "TDP restored by the end reset: $(tdp_fmt "$now") (start: $(tdp_fmt "$TDP_START"))"; rm -f "$TDP_MARKER"; fi
  return 0
}
tdp_exit() {  # the EXIT trap: a run that ended before tdp_end (an error, INT, TERM)
  [ "${TDP_DONE:-0}" = 0 ] || return 0
  TDP_DONE=1
  if [ "${TDP_SET:-0}" != 1 ] || tdp_refused > /dev/null; then tdp_skip "run ended early"; return 0; fi
  tdp_restore "run ended early" keep; return 0
}
