#!/usr/bin/env bash
# Single source of truth for "is this board usable, and which chips?". Emits shell assignments:
#
#   TT_HOLDERS          PIDs holding any /dev/tenstorrent/N (authoritative "busy")
#   TT_ACTIVE_HARNESS   harness PIDs that are burning CPU (starting up, about to grab a chip)
#   TT_ORPHAN_HARNESS   harness PIDs matching by name but holding nothing and idle -- NOT busy
#   TT_HEALTHY_CHIPS    comma-separated device ids whose ARC answers
#   TT_HEALTHY_COUNT    count of the above
#
# Callers: eval "$(.github/scripts/tt_device_state.sh)"
#
# Two lessons are encoded here, both cost real time on 2026-07-30:
#
# 1. Matching harness processes BY NAME alone is not a busy signal. Six tracy-capture processes
#    from a perf run that died four hours earlier held no device node and had accrued 16s of CPU
#    between them, yet made preflight report idle=false for four consecutive days. Because a
#    non-idle preflight skips `verify` and still concludes success, the Actions page showed four
#    green runs that verified nothing. Silence read as health.
# 2. Board health is PER CHIP, not per board. `tt-smi -s` hangs outright when any chip's ARC is
#    down, so a board with one good chip and three dead ones looked entirely unusable -- when the
#    gates in fact pass on the good chip once pinned to it. Read each chip's telemetry from sysfs
#    instead: a dead ARC reports tt_serial=FFFFFFFFFFFFFFFF and tt_card_type=unknown.
set -u

HARNESS_RE='agent\.loop|agent\.before_loop|tt_hw_planner|tracy-capture'

holders=""
for d in /dev/tenstorrent/[0-9]*; do
  [ -e "$d" ] || continue
  h=$(fuser "$d" 2>/dev/null | tr -s ' ') || true
  [ -n "$h" ] && holders="$holders $h"
done
holders=$(echo "$holders" | tr -s ' ' '\n' | grep -E '^[0-9]+$' | sort -u | tr '\n' ' ')

# CPU-time delta separates a harness that is starting up from one that died and never reaped.
cpu_ticks() {
  awk '{print $14 + $15}' "/proc/$1/stat" 2>/dev/null || echo 0
}

# Never match ourselves or an ancestor. A caller that merely MENTIONS these names -- a step that
# prints the reap command, say -- otherwise shows up as a harness, and a busy one at that if it
# happens to be burning CPU. Self-matching pgrep is how an earlier waiter in this repo spun forever.
selfchain=" $$ "
p=$$
while [ "$p" -gt 1 ]; do
  p=$(awk '{print $4}' "/proc/$p/stat" 2>/dev/null) || break
  [ -n "${p:-}" ] || break
  selfchain="$selfchain$p "
done

cand=$(pgrep -f "$HARNESS_RE" 2>/dev/null | sort -u || true)
cand=$(for c in $cand; do echo "$selfchain" | grep -q " $c " || echo "$c"; done)
before=""
for p in $cand; do before="$before $p:$(cpu_ticks "$p")"; done
[ -n "$cand" ] && sleep 2

active=""; orphan=""
for p in $cand; do
  kill -0 "$p" 2>/dev/null || continue
  was=$(echo "$before" | tr ' ' '\n' | awk -F: -v pid="$p" '$1==pid {print $2}')
  now=$(cpu_ticks "$p")
  if echo " $holders " | grep -q " $p "; then
    active="$active $p"
  elif [ "${now:-0}" -gt "${was:-0}" ]; then
    active="$active $p"
  else
    orphan="$orphan $p"
  fi
done

healthy=""
for d in /sys/class/tenstorrent/tenstorrent!*; do
  [ -e "$d" ] || continue
  id=${d##*!}
  serial=$(timeout 5 cat "$d/tt_serial" 2>/dev/null || echo "")
  ctype=$(timeout 5 cat "$d/tt_card_type" 2>/dev/null || echo "unknown")
  case "$serial" in
    ""|*[Ff][Ff][Ff][Ff][Ff][Ff][Ff][Ff]*) continue ;;
  esac
  [ "$ctype" = "unknown" ] && continue
  healthy="$healthy${healthy:+,}$id"
done

printf 'TT_HOLDERS="%s"\n' "$(echo "$holders" | xargs 2>/dev/null || true)"
printf 'TT_ACTIVE_HARNESS="%s"\n' "$(echo "$active" | xargs 2>/dev/null || true)"
printf 'TT_ORPHAN_HARNESS="%s"\n' "$(echo "$orphan" | xargs 2>/dev/null || true)"
printf 'TT_HEALTHY_CHIPS="%s"\n' "$healthy"
printf 'TT_HEALTHY_COUNT=%s\n' "$(echo "$healthy" | tr ',' '\n' | grep -c . || true)"
