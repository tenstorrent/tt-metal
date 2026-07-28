#!/bin/bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Blackhole Galaxy system diagnostic tool entrypoint.
# Thin wrapper: validate tier, set tt-metal env, hand off to diag_runner.py.
#
# Tiers:
#   light   ~5 min  snapshot validate + 1 PCI reset + GDDR train/BIST + eth link_up
#   medium          light + eth bandwidth + GDDR fast-pattern stress
#   deploy          3 resets (1x -r then 2x -glx_reset) + full GDDR patterns + eth bandwidth
#
# Designed to match tt-metal's run_upstream_tests_vanilla.sh shape.
# Can be used as the ENTRYPOINT of a docker image
# (TEST_COMMAND=tools/scaleout/exabox/healt_check_test_suite/run_diag.sh, CMD=light|medium|deploy).

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

show_help() {
    cat <<EOF
Usage: $0 {light|medium|deploy} [diag_runner.py options]

Tiers:
  light    Smoke check: snapshot validate + 1 PCI reset + GDDR train/BIST + eth link_up
  medium   light + eth bandwidth + GDDR fast-pattern stress
  deploy   3 resets + full GDDR pattern set + eth bandwidth

Forwarded options (see diag_runner.py --help for details):
  --dry-run              Print intended subprocess calls without executing destructive steps
  --input-snapshot PATH  Use pre-captured tt-smi snapshot JSON (skip live tt-smi call)
  --tt-smi-path PATH     Override tt-smi binary location
  --tt-metal-path PATH   Override tt-metal repo root (for unit_tests_deployment binary)
  --output PATH          Write JSON report to PATH (default: ./diag_report.json)
EOF
}

TIER="${1:-}"
case "$TIER" in
    light|medium|deploy) ;;
    -h|--help|"")
        show_help
        exit 0
        ;;
    *)
        echo "Error: unknown tier '$TIER'. Expected: light, medium, deploy" >&2
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
