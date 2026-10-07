#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Qwen3-VL e2e on the Quasar emu-quasar-2x3 RTL emulator (shared Zebu farm; needs your IRD reservation).
set -euo pipefail
shopt -s nullglob
# shellcheck source=_common.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

# The emulator runs at kHz; four hours bounds a 2+2 layer tiny run with margin.
TIMEOUT=14400
parse_args "$@"

export TT_METAL_SIMULATOR="${QWEN_EMU_DIR:-/proj_sw/user_dev/${USER}/tt-umd-simulators/build/emu-quasar-2x3/}"
if [[ ! -d "${TT_METAL_SIMULATOR}" ]]; then
  printf 'missing emulator dir %s\n' "${TT_METAL_SIMULATOR}" >&2
  exit 1
fi
if [[ -z "${NNG_SOCKET_ADDR:-}" ]]; then
  printf 'NNG_SOCKET_ADDR is not set (see the testing-with-quasar-emulator skill)\n' >&2
  exit 1
fi
export NNG_SOCKET_LOCAL_PORT="${NNG_SOCKET_LOCAL_PORT:-5555}"
if pgrep -u "${USER}" -f "pytest .*test_qwen3_vl_e2e" >/dev/null; then
  printf 'another qwen3_vl e2e pytest of yours is running; refusing to share the NNG port\n' >&2
  exit 1
fi
export MESH_DEVICE=N150
apply_debug_profile
trap 'printf "\nIf this run hung, check for leftover Zebu jobs (testing-with-quasar-emulator skill).\n"' EXIT
run_pytest "emu_2x3" --qwen-quasar-config --qwen-expect-grid "${EMU_GRID}"
