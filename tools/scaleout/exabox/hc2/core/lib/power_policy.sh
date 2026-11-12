# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# lib/power_policy.sh: the KMD power_policy
# value that run.sh's header reads read-only into TRAY_POWER_POLICY ('unknown' if unread). On one bad chip the numeric
# check caught its GDDR bank fault in 63 of 89 runs with power_policy=Y and in 0 of 15 with N, so a value other than
# Y (or 1) is a WARN, never a FAIL. run.sh changes the parameter only under the opt-in --kmd-power-policy (lib/kmd_policy.sh).
# Functions only; every mode calls the same ones.
pp_on() { case ${TRAY_POWER_POLICY:-unknown} in Y|y|1) return 0 ;; esac; return 1; }
# pp_warn: one WARN line when power_policy is not Y (or 1) or was not read; nothing under Y
pp_warn() {
  local v=${TRAY_POWER_POLICY:-unknown}
  if [ "$v" = unknown ]; then echo "WARN power_policy unread: DRAM error detection unknown"
  elif ! pp_on; then echo "WARN power_policy=$v: reduced DRAM error detection (GDDR bank faults may not show)"; fi
}
# pp_tag: 'power_policy=<v>' for the RESULT, run summary and repair lines; under any other value it says a pass here
# does not clear a known DRAM fault
pp_tag() {
  local v=${TRAY_POWER_POLICY:-unknown}
  if pp_on; then echo "power_policy=$v"
  elif [ "$v" = unknown ]; then echo "power_policy=unknown (DRAM error detection unknown: a pass does not clear a known DRAM fault)"
  else echo "power_policy=$v (reduced DRAM error detection: a pass does not clear a known DRAM fault)"; fi
}
