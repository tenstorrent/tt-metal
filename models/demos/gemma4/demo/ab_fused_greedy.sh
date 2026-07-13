#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# 1:1 A/B for GEMMA4_FUSED_GREEDY on QB2 (P150x4).
#
# Isolates ONE variable: fused greedy ON vs OFF.
# Everything else is pinned identical (no spec, DRAM-sharded, no bounded sliding).
#
# Uses the ci-1 demo case: short prompt, stop_at_eos=False, 512 generated tokens
# so both runs decode the same number of tokens (no early-EOS skew).
#
# Usage (from tt-metal root, after activating python_env):
#   bash models/demos/gemma4/demo/ab_fused_greedy.sh
#   bash models/demos/gemma4/demo/ab_fused_greedy.sh --case batch-1   # optional alt
#
# Logs: /tmp/gemma4_ab_fused_{off,on}.log
# Summary printed at the end.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_METAL_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$TT_METAL_ROOT"

CASE="ci-1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --case) CASE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--case ci-1|batch-1|long-context-64k]"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

export TT_METAL_HOME="$TT_METAL_ROOT"
export PYTHONPATH="$TT_METAL_ROOT"
export ARCH_NAME="${ARCH_NAME:-blackhole}"
export MESH_DEVICE="${MESH_DEVICE:-P150x4}"

if [[ -z "${HF_MODEL:-}" ]]; then
  export HF_MODEL=~/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/main
fi

# ── pinned common env (identical for both legs) ─────────────────────────────
unset GEMMA4_SPECULATIVE
unset GEMMA4_SAMPLE_ON_DEVICE
unset TT_CACHE_PATH
export GEMMA4_DRAM_SHARDED=1
export GEMMA4_BOUNDED_SLIDING=0
# ci-1 uses max_seq_len=8192 from the parametrize; don't override unless needed.
# For long-context-64k you may want: export GEMMA4_MAX_SEQ_LEN=32768
export GEMMA4_FUSED_GREEDY_TRACE=1

LOG_OFF="/tmp/gemma4_ab_fused_off.log"
LOG_ON="/tmp/gemma4_ab_fused_on.log"
PYTEST=(pytest models/demos/gemma4/demo/text_demo_v2.py -k "$CASE" -sv)

echo "============================================================"
echo " Gemma4 fused-greedy 1:1 A/B"
echo " case=$CASE  MESH_DEVICE=$MESH_DEVICE  HF_MODEL=$HF_MODEL"
echo " common: DRAM_SHARDED=1 BOUNDED_SLIDING=0 SPECULATIVE=unset"
echo "============================================================"

if [[ ! -d "$HF_MODEL" ]]; then
  echo "ERROR: HF_MODEL path does not exist: $HF_MODEL"
  exit 1
fi

run_leg() {
  local label="$1"
  local fused="$2"
  local log="$3"
  echo ""
  echo "── Leg: $label (GEMMA4_FUSED_GREEDY=$fused) ──"
  export GEMMA4_FUSED_GREEDY="$fused"
  # Fresh device state between legs reduces cross-run noise after hangs.
  if command -v tt-smi >/dev/null 2>&1; then
    tt-smi -r || true
    sleep 2
  fi
  "${PYTEST[@]}" 2>&1 | tee "$log"
  echo "── Leg $label done → $log ──"
}

extract_metrics() {
  local log="$1"
  # Prefer the fused steady-state line when present; else Decode: from metrics block.
  local fused_line decode_line gen_line
  fused_line="$(grep -E '\[plain-fused\]' "$log" | tail -1 || true)"
  decode_line="$(grep -E 'Decode: .*tok/s/user' "$log" | tail -1 || true)"
  gen_line="$(grep -E 'generated tokens:' "$log" | tail -1 || true)"
  echo "  generated: ${gen_line:-<missing>}"
  if [[ -n "$fused_line" ]]; then
    echo "  fused:     $fused_line"
  fi
  echo "  decode:    ${decode_line:-<missing>}"
  # Confirm which path ran
  if grep -q 'GEMMA4_FUSED_GREEDY=1' "$log"; then
    echo "  path:      FUSED (log confirms GEMMA4_FUSED_GREEDY=1)"
  elif grep -q 'Starting decode loop' "$log"; then
    echo "  path:      HOST (no fused banner)"
  else
    echo "  path:      UNKNOWN (check log)"
  fi
}

run_leg "OFF (host greedy baseline)" "0" "$LOG_OFF"
run_leg "ON  (fused greedy)"         "1" "$LOG_ON"

echo ""
echo "============================================================"
echo " RESULTS — compare Decode tok/s/user (ignore TTFT; fused uses eager prefill)"
echo "============================================================"
echo ""
echo "[OFF] host greedy  ($LOG_OFF)"
extract_metrics "$LOG_OFF"
echo ""
echo "[ON ] fused greedy ($LOG_ON)"
extract_metrics "$LOG_ON"
echo ""
echo "Fill-in table:"
echo "  | Leg   | ms/token | tok/s/user | generated | path  |"
echo "  | OFF   |          |            |           | host  |"
echo "  | ON    |          |            |           | fused |"
echo "  | Δ     |          |            |           |       |"
echo ""
echo "Sanity: generated token counts should match (ci-1 → 512)."
echo "If ON early-stops or hangs: tt-smi -r and re-run the ON leg only."
echo "============================================================"
