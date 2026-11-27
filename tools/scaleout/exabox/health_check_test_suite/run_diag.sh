#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Blackhole Galaxy system diagnostic tool entrypoint.
# Thin wrapper: validate tier, set tt-metal env, hand off to diag_runner.py.
#
# Tiers (what each one runs is defined in tiers.py; keep the case below and the
# --help text in step with it):
#   flash           quick ETH link health, the unit read as found: snapshot validate
#                   + two tt-triage check_eth_status reads 30 s apart
#                   + sys-triage bundle, if the host has the package and BMC credentials.
#                   No resets, no tests
#   daily           snapshot validate + 2 resets (-r, -glx_reset) + GDDR fast-pattern stress
#                   + eth link_up + eth bandwidth
#                   + post-test -glx_reset + triage (host_side + device_side)
#                   + QSFP tests (ETH link, cabling and module state), if the host has the package
#   weekly          3 resets (1x -r then 2x -glx_reset) + full GDDR patterns + eth link_up
#                   + eth bandwidth + didt matmul stress (pytest, galaxy mesh)
#                   + post-test -glx_reset + triage (host_side + device_side)
#                   + QSFP tests (ETH link, cabling and module state), if the host has the package
#   full            weekly + sys-triage bundle, if the host has the package and BMC credentials:
#                   everything, for the logs on a DIIM ticket
#   pre_bmc_reboot  data collection before a BMC reboot, no resets: snapshot
#                   + triage (host_side + device_side)
#                   + QSFP tests (ETH link, cabling and module state), if the host has the package
#                   + sys-triage bundle, if the host has the package and BMC credentials
#   post_bmc_reboot the same as pre_bmc_reboot for now, run after the BMC reboot
# The old names (light, medium, deploy, pre_reboot) still work; see DEPRECATED_TIER_ALIASES.
#
# Designed to match tt-metal's run_upstream_tests_vanilla.sh shape.
# Can be used as the ENTRYPOINT of a docker image
# (TEST_COMMAND=tools/scaleout/exabox/health_check_test_suite/run_diag.sh, CMD=<tier>).

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

show_help() {
    cat <<EOF
Usage: $0 {flash|daily|weekly|full|pre_bmc_reboot|post_bmc_reboot} [diag_runner.py options]

Tiers:
  flash           Quick ETH link health, the unit read as found: snapshot validate + two
                  tt-triage check_eth_status reads 30 s apart + sys-triage. No resets,
                  no tests.
  daily           2 resets + GDDR fast-pattern stress + eth link_up + eth bandwidth
                  + post-test reset + triage + QSFP tests
  weekly          3 resets + full GDDR pattern set + eth link_up + eth bandwidth
                  + didt matmul stress (pytest) + post-test reset + triage + QSFP tests
  full            weekly + sys-triage: everything, for the logs on a DIIM ticket
  pre_bmc_reboot  Data collection before a BMC reboot: snapshot + triage + QSFP tests
                  + sys-triage. No resets, no tests.
  post_bmc_reboot The same as pre_bmc_reboot for now, run after the BMC reboot.

The old names still work and print a deprecation warning: light and medium run daily,
deploy runs weekly, pre_reboot runs pre_bmc_reboot.

The ETH status read (flash) runs tools/triage/triage.py from this checkout under
python_env/bin/python (create_venv.sh installs the triage requirements there), else
the python running this script. It checks PCIe in sysfs first and does not open the
chips if one has dropped off; the raw reads land in <output dir>/logs/eth_status_read_*.

The QSFP tests check ETH link training, cabling and module state across all 448
ports. They run on every tier but flash when \`tt-bh-glx-cluster-debug\` is on PATH
(it ships in the syseng cluster-debug .deb); a host without the package skips the
phase and says so. Nothing needs configuring to turn them on.

sys-triage collects the syseng system and per-tray triage bundle (FW stack, FRU, CPLD
DIAG, lspci, kernel logs, per-chip vuart) into <output dir>/logs/sys_triage/, as the last
step of flash, full, pre_bmc_reboot and post_bmc_reboot only. It runs when \`sys-triage\` is on
PATH (same .deb) and the BMC credentials are set; otherwise the phase reports SKIP with
the reason. It never gates the run.

Forwarded options (see diag_runner.py --help for details):
  --dry-run              Print intended subprocess calls without executing destructive steps
  --input-snapshot PATH  Use pre-captured tt-smi snapshot JSON (skip live tt-smi call)
  --tt-smi-path PATH     Override tt-smi binary location
  --tt-metal-path PATH   Override tt-metal repo root (for unit_tests_deployment binary)
  --output PATH          Write JSON report to PATH (default: ./diag_report.json)
  --triage-dir PATH      Triage scripts dir (default: \$HC_TRIAGE_DIR, else tools/scaleout/kmd_triage)
  --skip-triage          Skip the post-test reset and the triage phase
  --triage-gating        Let triage FAILs gate the run (default: held at WARN)
  --skip-eth-status      Skip the ETH status read
  --skip-qsfp-tests      Skip the QSFP tests. Named for the phase, not the collector's own
                         --skip-qsfp, which drops the cage sweep but still collects ETH
  --qsfp-tool-path PATH  Override the tt-bh-glx-cluster-debug binary
  --qsfp-gating          Report QSFP findings at their real severity and let them gate
                         the run. Off by default while the tool is validated: findings
                         are recorded in full but every check reports PASS
  --qsfp-descriptor PATH Optional. factory_system_descriptor.textproto, which adds an expected
                         partner for the cage-attached links. Without one the 104 soldered
                         internal links are still checked against the collector's built-in
                         topology table; only the cabling checks narrow, and they say so
                         rather than reporting coverage they don't have
  --skip-sys-triage      Skip the sys-triage bundle
  --sys-triage-path PATH Override the sys-triage binary
  --bmc-ip IP            BMC address for sys-triage (default: \$BMC_IP)
  --bmc-user USER        BMC user for sys-triage (default: \$BMC_USER)
  --bmc-password PASS    BMC password for sys-triage (default: \$BMC_PASSWORD, preferred:
                         a flag value is visible in the process list)
EOF
}

TIER="${1:-}"
case "$TIER" in
    flash|daily|weekly|full|pre_bmc_reboot|post_bmc_reboot|light|medium|deploy|pre_reboot) ;;
    -h|--help|"")
        show_help
        exit 0
        ;;
    *)
        echo "Error: unknown tier '$TIER'. Expected: flash, daily, weekly, full, pre_bmc_reboot, post_bmc_reboot" >&2
        show_help
        exit 1
        ;;
esac
shift

# tt-metal env. Respect existing values when set; default to repo root.
export TT_METAL_HOME="${TT_METAL_HOME:-$SCRIPT_DIR/../../../..}"
export PYTHONPATH="${PYTHONPATH:-$TT_METAL_HOME}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-$TT_METAL_HOME/build/lib}"
# tt-smi often lives at ~/.local/bin or /usr/local/bin but isn't on PATH
# in non-login SSH sessions.
export PATH="$PATH:$HOME/.local/bin:/usr/local/bin"

exec python3 "$SCRIPT_DIR/diag_runner.py" --tier "$TIER" "$@"
