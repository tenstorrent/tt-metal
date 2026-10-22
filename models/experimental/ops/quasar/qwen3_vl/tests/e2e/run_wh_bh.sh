#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# WH/BH baseline with the Quasar config: same flags as the Quasar scripts, on ttsim (--ttsim wh|bh) or hardware.
set -euo pipefail
shopt -s nullglob
TARGET_USAGE="[--ttsim wh|bh] [--grid 2x3|native] [--config quasar|native]"
# shellcheck source=_common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

TTSIM=""
# 2x3 reproduces the emulator's compute grid so the baseline runs the same configs; native uses the full chip.
GRID=2x3
# quasar: bf16 Quasar config (what Quasar runs); native: the unmodified bf8 config, as a hardware control.
CONFIG=quasar
TIMEOUT=14400
parse_target_flag() {
  case "$1" in
    --ttsim) TTSIM="$2" ;;
    --grid) GRID="$2" ;;
    --config) CONFIG="$2" ;;
    *) return 1 ;;
  esac
}
parse_args "$@"

if [[ -n "${TTSIM}" ]]; then
  export TT_METAL_SIMULATOR="${QWEN_TTSIM_DIR:-/localdev/${USER}/ttsim}/sim_${TTSIM}/libttsim.so"
  if [[ ! -f "${TT_METAL_SIMULATOR}" ]]; then
    printf 'missing %s (stage it, see README)\n' "${TT_METAL_SIMULATOR}" >&2
    exit 1
  fi
else
  unset TT_METAL_SIMULATOR
fi
export MESH_DEVICE=N150
apply_debug_profile
GRID_ARGS=()
if [[ "${GRID}" == 2x3 ]]; then
  # WH/BH compute grids start at logical (0,0), so end (1,0) gives the emulator's 2x1.
  export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="1,0"
  GRID_ARGS=(--qwen-expect-grid "${EMU_GRID}")
fi
CONFIG_ARGS=()
if [[ "${CONFIG}" == quasar ]]; then CONFIG_ARGS=(--qwen-quasar-config); fi
run_pytest "wh_bh_${TTSIM:-hw}_${GRID}_${CONFIG}" "${CONFIG_ARGS[@]}" "${GRID_ARGS[@]}"
