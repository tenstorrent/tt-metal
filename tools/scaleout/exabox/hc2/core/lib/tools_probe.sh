# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/tools_probe.sh: locate the on-hang chip-state tools (tt-triage, ttexalens) once, at preflight.
# Functions only. tools_locate <python> sets and exports SNAP_TRIAGE and HEAVY_TRIAGE (when found), CS_TRIAGE (path or ""),
# CS_EXALENS (1/0) and prints one WARN line when a tool is missing. Reads only: no device is opened.
# The triage command must run under the run's own venv python (the one run.sh uses, $PY), which has
# ttexalens; a bare system python3 lacks ttexalens on the H3/quad hosts and the capture dies with
# "Module 'ttexalens' not found" (seen in a hold read's triage_noc0.txt). The triage command embeds the venv python
# absolute path so the hold attach lines hold.txt hands out run with it, on the host or from another host.
# The venv pythons searched for ttexalens, in order. TRAY_VENV_SEARCH (space-separated paths or globs)
# replaces the default list, so tests do not pick up another user's venv on the host they run on.
tools_venv_candidates() {
  local c
  if [ -z "${TRAY_VENV_SEARCH:-}" ]; then
    printf '%s\n' "$HOME"/venv/bin/python "$HOME"/venv3/bin/python; return
  fi
  # shellcheck disable=SC2086  # word-split and glob-expand the list on purpose
  for c in $TRAY_VENV_SEARCH; do printf '%s\n' "$c"; done
}
tools_locate() {
  local py=$1 t="" c venvpy=""
  if [ -n "${SNAP_TRIAGE:-}" ]; then t=$SNAP_TRIAGE
  elif [ -n "${HEAVY_TRIAGE:-}" ]; then t=$HEAVY_TRIAGE
  elif [ -n "${UB0_TRIAGE:-}" ]; then t=$UB0_TRIAGE
  else
    for c in "${TT_METAL_HOME:-/nonexistent}" /home/user/tt-metal "$HOME/tt-metal"; do
      [ -f "$c/tools/triage/triage.py" ] && { t="python3 $c/tools/triage/triage.py"; export TT_METAL_HOME=${TT_METAL_HOME:-$c}; break; }
    done
    [ -n "$t" ] || { c=$(command -v tt-triage 2> /dev/null) && t=$c; }
    [ -n "$t" ] || { [ -x "$(dirname "$py")/tt-triage" ] && t=$(dirname "$py")/tt-triage; }
  fi
  # the triage script needs ttexalens; run it with a python that has it. Prefer the run's own
  # venv python ($PY, i.e. the tools_locate argument), then $HOME/venv and $HOME/venv3.
  if [ -n "$t" ] && [ -n "${TT_METAL_HOME:-}" ] && [ -f "$TT_METAL_HOME/tools/triage/triage.py" ]; then
    case $t in python3\ *) venvpy=""
      [ -x "$py" ] && "$py" -I -c 'import ttexalens' > /dev/null 2>&1 && venvpy=$py
      [ -z "$venvpy" ] && while IFS= read -r c; do
        [ -x "$c" ] || continue
        "$c" -I -c 'import ttexalens' > /dev/null 2>&1 && { venvpy=$c; break; }
      done < <(tools_venv_candidates) 2>/dev/null
      [ -n "$venvpy" ] && t="$venvpy $TT_METAL_HOME/tools/triage/triage.py"
      [ -z "$venvpy" ] && echo "WARN: no python with ttexalens found for triage: trying system python3" >&2
      ;;
    esac
  fi
  CS_TRIAGE=$t; CS_EXALENS=0
  "$py" -I -c 'import ttexalens' > /dev/null 2>&1 && CS_EXALENS=1
  if [ -n "$t" ]; then export SNAP_TRIAGE=$t HEAVY_TRIAGE=$t; fi
  export CS_TRIAGE CS_EXALENS
  local miss=""
  [ -n "$t" ] || miss="tt-triage"
  [ "$CS_EXALENS" = 1 ] || miss="$miss${miss:+ and }ttexalens"
  [ -z "$miss" ] || echo "WARN: $miss not found: on-hang chip state will not be captured"
}
# a python that can import ttexalens, for the triage script's fallback paths (hold.sh, noc_snapshot.sh).
# Order: the run's own venv python, then tools_venv_candidates ($HOME/venv, $HOME/venv3); "" if none.
tools_venv_python() {
  local c py=${1:-${PY:-python3}}
  [ -x "$py" ] && "$py" -I -c 'import ttexalens' > /dev/null 2>&1 && { echo "$py"; return; }
  while IFS= read -r c; do
    [ -x "$c" ] || continue
    "$c" -I -c 'import ttexalens' > /dev/null 2>&1 && { echo "$c"; return; }
  done < <(tools_venv_candidates) 2>/dev/null
  return 1
}
tools_captured_txt() {  # the claim for a hang's fault line: "every chip captured" only when both tools were found
  if [ -n "${CS_TRIAGE:-}" ] && [ "${CS_EXALENS:-0}" = 1 ]; then echo "every chip captured"
  else echo "chip state not captured (no triage/exalens)"; fi
}
