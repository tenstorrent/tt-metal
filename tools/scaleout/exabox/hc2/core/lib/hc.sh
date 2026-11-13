# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# shellcheck shell=bash disable=SC1090,SC2034  # sources a hook path; the off stubs set variables their callers read
# lib/hc.sh: the core's hooks for local features, off by default. A caller turns one on by setting its variable:
#   HC_SETUP         a command that prepares and checks the Python env (run in the code dir). Unset: the core only
#                    checks that 'python3 -c "import ttnn"' works and runs python3.
#   HC_HOLDFILE      a file sourced for hf_check (a held host runs nothing, NOT RUN exit 10). Unset: no holds.
#   HC_HOLD          a file sourced for --hold (hold_init, hold_point, hold_done). Unset: --hold is refused with a line.
#   HC_BROKER_YIELD  a file sourced for by_start/by_check/by_stop (give the device up to a queued job). Unset: never yields.
#   HC_VENV          the venv HC_SETUP makes (its bin/python runs the python stages). Unset: ./venv.
#   HC_SITE_FILE     ':'-separated key=value files of local site data (hc_site <key>). Unset: none.
# Set to an empty value, a hook stays off. A local wrapper may set them all in lib/hc_hooks.sh; this file sources it
# once when it exists (a tree without it runs the core alone).
if [ -z "${HC_LOADED:-}" ]; then
  HC_LOADED=1
  hc_hooks=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hc_hooks.sh
  if [ -r "$hc_hooks" ]; then . "$hc_hooks"; fi
  unset hc_hooks
fi
# hc_site <key>: the key's value in the first HC_SITE_FILE file that has it; "" when none does
hc_site() {
  local f v IFS=:
  for f in ${HC_SITE_FILE:-}; do
    [ -r "$f" ] || continue
    v=$(sed -n "s/^$1=//p" "$f" | head -1)
    if [ -n "$v" ]; then printf '%s\n' "$v"; return 0; fi
  done
  return 0
}
# the no-op stand-ins a hook that is off leaves in place
hc_holdfile_off() { HF_WHY=""; hf_check() { HF_WHY=""; return 0; }; }
hc_hold_off() {
  HOLD_ACTIVE=0; HOLD_USED=""
  hold_init() { if [ "${HOLD:-0}" = 1 ]; then echo "--hold refused: no hold hook (HC_HOLD unset); the run goes on without --hold"; HOLD=0; fi; return 0; }
  hold_point() { return 0; }
  hold_done() { return 0; }
  hold_broker() { return 1; }
}
hc_broker_yield_off() {
  BY_PID=""; FL_YIELD=""; FL_YIELD_AT=""
  by_file() { echo "${FL_OUT:-${OUT:-.}}/yield.txt"; }
  by_on() { return 1; }
  by_start() { return 0; }
  by_check() { return 0; }
  by_stop() { return 0; }
}
