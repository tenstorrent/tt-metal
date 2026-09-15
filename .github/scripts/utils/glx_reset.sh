#!/usr/bin/env bash
# Reset a Galaxy (6U) cluster from the runner host.
#
# Two host-specific details this wraps:
#  - tt-smi lives in the runner provisioning venv, not on the default PATH.
#  - The per-board PCIe reset (tt-smi -r) is invalid on topology-6u runners and
#    fails instantly; the galaxy reset is -glx_reset_auto (matching the
#    topology-6u branch of TT_SMI_RESET_COMMAND in ttnn-run-sweeps.yaml).
#
# Kept as its own script so activating the venv cannot leak into the calling
# step's environment (which needs the setup-python ttnn env for python3).
set -uo pipefail

ACTIVATE=/opt/tt_metal_infra/provisioning/provisioning_env/bin/activate
if [ -f "$ACTIVATE" ]; then
  # shellcheck disable=SC1090,SC1091
  source "$ACTIVATE"
fi

exec tt-smi -glx_reset_auto
