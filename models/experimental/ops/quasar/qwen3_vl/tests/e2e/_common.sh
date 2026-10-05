#!/usr/bin/env bash
# shellcheck shell=bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Shared flag parsing and run-folder handling for the Qwen3-VL e2e run scripts (sourced, not executed).
set -euo pipefail
shopt -s nullglob

E2E_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${E2E_DIR}/../../../../../../.." && pwd)"
TEST_PATH="models/experimental/ops/quasar/qwen3_vl/tests/e2e/test_qwen3_vl_e2e.py"

# The emu-quasar-2x3 build exposes a 2x1 compute grid (tt_metal/core_descriptors/quasar_simulation_2x3_arch.yaml).
# The override is the grid's end coordinate, so its value depends on where each target's compute grid starts.
EMU_GRID=2x1
export EMU_GRID

# Two vision blocks and two text layers keep emulator runs near an hour while covering a block-to-block handoff.
VISION_LAYERS=2
TEXT_LAYERS=2
# One decode step exercises the paged-KV decode path once; more steps repeat the same ops.
DECODE_STEPS=1
# Tap deepstack at block 0 so two vision blocks still run the deepstack mergers and their add into the text layers.
DEEPSTACK_AT=0
# Tiny (256x256 image, 128-token prefill) is the fast iteration size; demo matches the graph captures.
SIZE=tiny
KV_BLOCKS="" # paged KV-cache blocks of 32 tokens; empty uses the preset's value.
HOST_OPS=""
DISABLE_WA=""
# The default profile catches LLK asserts via watcher without NoC sanitize, which is 20-30x slower.
DEBUG_PROFILE=default
NOC_SANITIZE=0
TIMEOUT=""
EXTRA_PYTEST=()

usage() {
  printf '%s\n' "Usage: $0 [--size tiny|demo] [--vision-layers N] [--text-layers N] [--decode-steps K]" \
    "  [--deepstack-at I|real] [--kv-blocks N] [--host-ops a,b|all] [--disable-wa a,b]" \
    "  [--debug fast|default|deep] [--noc-sanitize] [--fp32-dest-acc] [--timeout S] ${TARGET_USAGE:-} [-- <pytest args>]"
}

# Parses all flags. Target scripts may define parse_target_flag "<flag>" "<value>" (returns 0 if consumed);
# every target flag takes exactly one value.
parse_args() {
  while (($#)); do
    case "$1" in
      --size) SIZE="$2" ;;
      --vision-layers) VISION_LAYERS="$2" ;;
      --text-layers) TEXT_LAYERS="$2" ;;
      --decode-steps) DECODE_STEPS="$2" ;;
      --deepstack-at) DEEPSTACK_AT="$2" ;;
      --kv-blocks) KV_BLOCKS="$2" ;;
      --host-ops) HOST_OPS="$2" ;;
      --disable-wa) DISABLE_WA="$2" ;;
      --debug) DEBUG_PROFILE="$2" ;;
      --timeout) TIMEOUT="$2" ;;
      --noc-sanitize)
        NOC_SANITIZE=1
        shift
        continue
        ;;
      --fp32-dest-acc)
        # A/B on hardware: keep fp32 dest accumulation (undefined on ttsim WH, QUASAR_GAPS S1).
        export QWEN_QSR_FP32_DEST_ACC=1
        shift
        continue
        ;;
      --)
        shift
        EXTRA_PYTEST=("$@")
        return 0
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        if ! declare -F parse_target_flag >/dev/null || ! parse_target_flag "$1" "${2:-}"; then
          printf 'unknown flag %s\n' "$1" >&2
          usage
          exit 1
        fi
        ;;
    esac
    shift 2
  done
}

apply_debug_profile() {
  export TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 TT_METAL_DISABLE_SFPLOADMACRO=1
  # ttnn op hooks (progress.log) only fire when fast runtime mode is off.
  export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}'
  case "${DEBUG_PROFILE}" in
    fast) ;;
    default | deep)
      export TT_METAL_WATCHER=1 TT_METAL_WATCHER_TEST_MODE=1 TT_METAL_LLK_ASSERTS=1
      if [[ "${NOC_SANITIZE}" -eq 0 ]]; then export TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1; fi
      if [[ "${DEBUG_PROFILE}" == deep ]]; then
        export TT_METAL_WATCHER_DUMP_ALL=1 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_DPRINT_ONE_FILE_PER_RISC=1 \
          TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1 TT_METAL_WATCHER_DISABLE_PAUSE=1 TT_METAL_LOGGER_LEVEL=DEBUG
      fi
      ;;
    *)
      printf 'unknown --debug %s\n' "${DEBUG_PROFILE}" >&2
      exit 1
      ;;
  esac
}

# $1 = target name; remaining args = target-specific pytest options. Exits 0 PASS, 3 DIAGNOSTIC, else failure.
run_pytest() {
  local target="$1"
  shift
  local run_dir
  run_dir="${REPO_ROOT}/generated/qwen3_vl_quasar/${target}/$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p -- "${run_dir}"
  local cmd=(pytest "${TEST_PATH}" -sv "--timeout=${TIMEOUT}" --qwen-run-dir "${run_dir}"
    --qwen-size "${SIZE}" --qwen-vision-layers "${VISION_LAYERS}" --qwen-text-layers "${TEXT_LAYERS}"
    --qwen-decode-steps "${DECODE_STEPS}" --qwen-host-ops "${HOST_OPS}" --qwen-disable-wa "${DISABLE_WA}" "$@")
  if [[ "${DEEPSTACK_AT}" != real ]]; then cmd+=(--qwen-deepstack-at "${DEEPSTACK_AT}"); fi
  if [[ -n "${KV_BLOCKS}" ]]; then cmd+=(--qwen-kv-blocks "${KV_BLOCKS}"); fi
  cmd+=("${EXTRA_PYTEST[@]}")
  printf '%q ' "${cmd[@]}" >"${run_dir}/command.txt"
  env | grep -E '^(TT_|TTNN_|MESH_|HF_MODEL|NNG_|ARCH_|QWEN_)' | sort >"${run_dir}/env.txt" || true
  {
    git -C "${REPO_ROOT}" rev-parse HEAD
    git -C "${REPO_ROOT}" log --oneline origin/main..HEAD
  } >"${run_dir}/git.txt"
  local rc=0
  (cd -- "${REPO_ROOT}" && "${cmd[@]}") 2>&1 | tee "${run_dir}/run.log" || rc=$?
  printf '\nRun folder: %s\n' "${run_dir}"
  if [[ -f "${run_dir}/pcc.md" ]]; then cat -- "${run_dir}/pcc.md"; fi
  if [[ -f "${run_dir}/progress.log" ]]; then
    printf '\nLast ops:\n'
    tail -n 5 -- "${run_dir}/progress.log"
  fi
  if [[ -f "${run_dir}/verdict.txt" ]] && grep -qx DIAGNOSTIC "${run_dir}/verdict.txt"; then exit 3; fi
  exit "${rc}"
}
