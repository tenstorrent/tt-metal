#!/usr/bin/env bash
# Shared paths, environment, and suite inventory for Quasar perf runners.

PERF_SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLK_ROOT="${LLK_ROOT:-$(cd "${PERF_SUITE_DIR}/../../.." && pwd)}"
CURRENT_USER="${USER:-$(id -un)}"
UMD_SIM_ROOT="${UMD_SIM_ROOT:-/proj_sw/user_dev/${CURRENT_USER}/tt-umd-simulators}"

export LLK_ROOT
export CHIP_ARCH="${CHIP_ARCH:-quasar}"
export TT_METAL_SIMULATOR="${TT_METAL_SIMULATOR:-${UMD_SIM_ROOT}/build/emu-quasar-1x3}"
export TT_UMD_SIMULATOR_PATH="${TT_UMD_SIMULATOR_PATH:-${TT_METAL_SIMULATOR}}"
export NNG_SOCKET_LOCAL_PORT="${NNG_SOCKET_LOCAL_PORT:-5555}"

PERF_SUITE_TESTS=(
  "02:perf_eltwise_unary_datacopy_quasar.py"
  "03:perf_eltwise_binary_broadcast_quasar.py"
  "04:perf_eltwise_binary_quasar.py"
  "05:perf_unpack_tilize_quasar.py"
  "06:perf_unpack_unary_operand_quasar.py"
  "07:perf_transpose_dest_quasar.py"
  "08:perf_pack_quasar.py"
  "09:perf_pack_untilize_quasar.py"
  "10:perf_unary_broadcast_quasar.py"
  "11:perf_pack_l1_acc_quasar.py"
  "12:perf_reduce_quasar.py"
  "13:perf_eltwise_binary_reuse_dest_quasar.py"
  "14:perf_unpack_reduce_col_tilizeA_strided_quasar.py"
)

prepare_quasar_perf_environment() {
  if [[ -z "${NNG_SOCKET_ADDR:-}" ]]; then
    echo "ERROR: set NNG_SOCKET_ADDR to the active reservation endpoint" >&2
    return 2
  fi
  if [[ ! -f "${LLK_ROOT}/tests/.venv/bin/activate" ]]; then
    echo "ERROR: LLK test environment not found: ${LLK_ROOT}/tests/.venv" >&2
    return 2
  fi

  # shellcheck source=/dev/null
  source "${LLK_ROOT}/tests/.venv/bin/activate"
}

install_instrumented_launcher() {
  local launcher="${INSTRUMENTED_LAUNCHER:-${UMD_SIM_ROOT}/emu/quasar-1x3/quasar-1x3_run_dev.instrumented.sh}"
  local target="${TT_METAL_SIMULATOR}/quasar-1x3_run_dev.sh"

  if [[ ! -f "$launcher" ]]; then
    echo "INFO: instrumented launcher not found; using existing simulator launcher: ${launcher}" >&2
    return 0
  fi
  if [[ ! -d "$TT_METAL_SIMULATOR" ]]; then
    echo "ERROR: simulator build directory not found: ${TT_METAL_SIMULATOR}" >&2
    return 2
  fi

  if [[ -f "$target" && ! -f "${target}.pre_instrument" ]]; then
    cp -a "$target" "${target}.pre_instrument"
    echo "Backed up simulator launcher to ${target}.pre_instrument"
  fi
  cp -a "$launcher" "$target"
  echo "Installed instrumented simulator launcher at ${target}"
}
