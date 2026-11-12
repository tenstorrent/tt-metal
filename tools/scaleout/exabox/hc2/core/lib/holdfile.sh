# lib/holdfile.sh: no neighbour or site checks; the only rule outside this host that stops a run or a reset is the
# user's own hold and allowlist data file. Its path: TRAYTEST_HOLD_FILE, else ${XDG_CONFIG_HOME:-~/.config}/tray-test/holds
# when that exists; no file, no check. One row per line, whitespace separated, '#' starts a comment:
#   hold  <host|*> <cite> [why]   this host takes no run and no reset
#   allow <host|*> <cite> [why]   with any allow row, only the hosts listed may run
#   no-cycle <host|*> <cite> [why]   this host takes no power cycle (reliability.sh's cycle()); runs and resets go on
# <cite> names the user message the row comes from and holds '#<digits>' (e.g. msg#5497). A hold row with no cite still
# holds (with a WARN); an allow row with no cite is ignored (with a WARN). <host> matches the short or full hostname,
# case-insensitive. A file named but unreadable counts as held. Never a hardcoded host list.
# hf_check [cycle]: 0 when this host may run; else 1 with HF_WHY. With 'cycle': 0 when it may also be power cycled (a
# no-cycle row refuses it too). WARN lines go to stderr. HF_HOST overrides the hostname (tests).
hf_path() {
  if [ -n "${TRAYTEST_HOLD_FILE:-}" ]; then echo "$TRAYTEST_HOLD_FILE"; return 0; fi
  local d=${XDG_CONFIG_HOME:-${HOME:-/nonexistent}/.config}/tray-test/holds
  [ -e "$d" ] && echo "$d"; return 0
}
hf_check() {
  local f s l k h c w hs hl nallow=0 ok=0 held="" sc=${1:-}
  HF_WHY=""; f=$(hf_path); [ -n "$f" ] || return 0
  if ! [ -r "$f" ] || ! s=$(cat -- "$f" 2>/dev/null); then HF_WHY="hold file $f unreadable (counts as held)"; return 1; fi
  hs=${HF_HOST:-$(hostname -s 2>/dev/null)}; hl=${HF_HOST:-$(hostname -f 2>/dev/null || hostname)}
  hs=${hs,,}; hl=${hl,,}
  while IFS= read -r l; do
    l=${l%%#[!0-9]*}; l=${l%#}; [[ $l == \#* ]] && continue   # a lone trailing '#' is a comment too
    read -r k h c w <<< "$l"; [ -n "${k:-}" ] || continue
    case ${k,,} in hold|allow) ;; no-cycle) [ "$sc" = cycle ] || continue ;; *) echo "WARN hold file $f: unknown row kind '$k' ignored" >&2; continue ;; esac
    if [[ ${c:-} != *\#[0-9]* ]]; then
      if [ "${k,,}" = allow ]; then echo "WARN hold file $f: allow row for '${h:-}' cites no user message; ignored" >&2; continue; fi
      echo "WARN hold file $f: ${k,,} row for '${h:-}' cites no user message; it still holds" >&2; w="${c:+$c }${w:-}"; c="no cite"
    fi
    [ "${k,,}" = allow ] && nallow=$((nallow + 1))
    case ${h,,} in '*'|"$hs"|"$hl") ;; *) continue ;; esac
    case ${k,,} in hold) held="hold on ${h} (${c}${w:+: ${w% }})" ;; no-cycle) [ -n "$held" ] || held="no-cycle hold on ${h} (${c}${w:+: ${w% }})" ;; *) ok=1 ;; esac
  done <<< "$s"
  [ -n "$held" ] && { HF_WHY="$held, hold file $f"; return 1; }
  [ "$nallow" -gt 0 ] && [ "$ok" = 0 ] && { HF_WHY="$hs is not on the allowlist in hold file $f"; return 1; }
  return 0
}
