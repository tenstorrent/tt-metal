#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Qwen3-VL e2e on craq-sim (Quasar functional simulator); emulator-sized grid by default, 8x4 on request.
set -euo pipefail
shopt -s nullglob
TARGET_USAGE="[--grid 2x3|8x4]"
# shellcheck source=_common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

# 2x3 matches the emulator so craq-sim is a fast pre-flight for emulator runs; 8x4 is craq-sim's full grid.
GRID=2x3
TIMEOUT=3600
# The watcher slows craq-sim ~150-300x per op (a 2.5 s add took 736 s); use --debug default only to localize hangs.
DEBUG_PROFILE=fast
parse_target_flag() {
  case "$1" in
    --grid) GRID="$2" ;;
    *) return 1 ;;
  esac
}
parse_args "$@"

export TT_METAL_SIMULATOR="${QWEN_CRAQ_SIM:-/localdev/${USER}/sim/libttsim.so}"
if [[ ! -f "${TT_METAL_SIMULATOR}" ]]; then
  printf 'missing craq-sim %s\n' "${TT_METAL_SIMULATOR}" >&2
  exit 1
fi
export MESH_DEVICE=N150
apply_debug_profile
GRID_ARGS=(--qwen-expect-grid 8x4)
if [[ "${GRID}" == 2x3 ]]; then
  # craq-sim's compute grid starts at logical (2,2), so end (3,2) gives the emulator's 2x1.
  export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2"
  GRID_ARGS=(--qwen-expect-grid "${EMU_GRID}")
fi
run_pytest "craq_${GRID}" "${GRID_ARGS[@]}"
