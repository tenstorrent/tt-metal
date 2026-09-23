#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Reproducible runner for bench_e2e_p150.py: pins every QWEN36_*/QWEN_GDN_*/QWEN_* env flag
# explicitly (so results never depend on a stale shell), checks the device is free, and runs the
# benchmark under a 30-minute watchdog. See ../REPRODUCE_P150_PERF.md.
#
# Usage: run_bench_e2e_p150.sh [f12|f13] [isl] [osl] [runs]
#   f12 (default) - phased chunk-parallel GDN prefill, this branch (atupe/qwen35-2b-p150-prefill-perf).
#   f13            - experimental fused FLA prim (QWEN_GDN_PATH=fused). Only works in a tree that
#                    has the fused prim built in (see the fused_fla_available check below); refuses
#                    to run otherwise. As of this branch, that means the fla-fused-eval branch/worktree.
#   isl  (default 4096) - or the literal "demo": passes --demo-prompt to bench_e2e_p150.py
#                          instead of --isl, and names the output file
#                          <config>_demo_osl<osl>_<timestamp>.json.
#   osl  (default 8)
#   runs (default 5)
set -euo pipefail

CONFIG="${1:-f12}"
ISL="${2:-4096}"
OSL="${3:-8}"
RUNS="${4:-5}"

case "$CONFIG" in
  f12|f13) ;;
  *) echo "usage: $(basename "$0") [f12|f13] [isl] [osl] [runs]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "$REPO_ROOT/python_env/bin/activate" ]; then
  echo "ERROR: $REPO_ROOT/python_env/bin/activate not found -- run ./create_venv.sh first (see REPRODUCE_P150_PERF.md)." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$REPO_ROOT/python_env/bin/activate"

echo "== repo: $REPO_ROOT =="
echo "== branch: $(git rev-parse --abbrev-ref HEAD) commit: $(git rev-parse HEAD) dirty_files: $(git status --porcelain | wc -l) =="

# ---------------------------------------------------------------------------------------------
# 1. Unset EVERY QWEN* var already in the shell, so nothing is inherited from a previous session.
# ---------------------------------------------------------------------------------------------
while IFS='=' read -r name _; do
  [ -n "$name" ] && unset "$name"
done < <(env | grep -E '^QWEN' || true)

# ---------------------------------------------------------------------------------------------
# 2. Core env.
# ---------------------------------------------------------------------------------------------
export HF_MODEL="Qwen/Qwen3.5-2B"
export MESH_DEVICE="P150"
export HF_HUB_OFFLINE=1
export TT_METAL_HOME="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

# ---------------------------------------------------------------------------------------------
# 3. Pin every QWEN36_*/QWEN35_*/QWEN9B_*/QWEN_* flag at its documented default (see the
# ALL_QWEN_FLAG_DEFAULTS table in bench_e2e_p150.py -- keep these two lists in sync; regenerate
# the flag NAMES with:
#   grep -h "os.environ.get(\"QWEN" -r models/demos/blackhole/qwen36 \
#       models/experimental/gated_attention_gated_deltanet | sed 's/.*environ.get(//' | cut -d, -f1 \
#       | sort -u
#
# A handful of flags are deliberately NOT exported here (left truly unset): they are read via a
# bare `if os.environ.get(X):` truthiness check, where exporting ANY non-empty string (even "0")
# would wrongly enable them, or their fallback is computed from other locals at call time (not a
# fixed literal), so a made-up literal here would silently override the correct computed default.
# Those are listed and explained at the bottom of this section.
# ---------------------------------------------------------------------------------------------
export QWEN35_GDN_DECODE_BF16=0
export QWEN35_GDN_STATE_BF16=0
export QWEN35_NO_REPEAT_NGRAM=0
export QWEN35_REP_PENALTY=1.0
export QWEN35_TEMP=0
export QWEN35_TOP_K=0
export QWEN35_TOP_P=1.0
export QWEN35_TP_DECODE_EAGER=0
export QWEN35_TP_PREFILL_EAGER=0

export QWEN36_ATTN_FUSED_QKV=1
export QWEN36_ATTN_GATE_FUSED=1
export QWEN36_ATTN_KV_BF8=0
export QWEN36_ATTN_L1_MAX_T=2048
export QWEN36_ATTN_QKNORM_HIFI2=1
export QWEN36_BATCHED_DECODE_MODE=shard
export QWEN36_BUCKET_TEST_CTX=8192
export QWEN36_BUCKET_TEST_WIDTHS=1,8
export QWEN36_CAPACITY_TEST_BMAX=8
export QWEN36_CAPACITY_TEST_EAGER_PROFILE=0
export QWEN36_CAPACITY_TEST_ITERS=20
export QWEN36_CAPACITY_TEST_TRIALS=5
export QWEN36_CAPACITY_TEST_WARMUP=3
export QWEN36_DEBUG_DECODE_TIMING=0
export QWEN36_DECODE_PROGCFG=1
export QWEN36_GDN_CONV_KDA=1
export QWEN36_GDN_CONV_KDA_FP32ACC=0
export QWEN36_GDN_CONV_LEGACY=0
export QWEN36_GDN_CONV_SILU_SHARDED=0
export QWEN36_GDN_CONV_T3_MAX=0
export QWEN36_GDN_CONV_TILED_SPLIT=0
export QWEN36_GDN_FUSED_PREFILL=1
export QWEN36_GDN_GATE_CLIP=0
export QWEN36_GDN_GATE_FUSED=1
export QWEN36_GDN_GB_LAYOUT=0
export QWEN36_GDN_L1_MAX_T=0
export QWEN36_GDN_NATIVE_CONV1D=1
export QWEN36_GDN_POST_L1=1
export QWEN36_GDN_POST_L1_OUTPROJ=1
export QWEN36_GDN_POST_L1_SCAN=1
export QWEN36_GDN_SPLIT_PROJ=1
export QWEN36_LAYER_L1_MAX_T=0
export QWEN36_LAYER_RESID_L1=0
export QWEN36_LMHEAD_MINIMAL=1
export QWEN36_LMHEAD_SPLIT=8
export QWEN36_MLP_FUSED_SWIGLU=1
export QWEN36_MLP_L1_OUT=1
export QWEN36_MLP_LEGACY_SHORT=0
export QWEN36_MLP_MINIMAL_MM=1
export QWEN36_PREFILL_DEBUG=0
export QWEN36_PREFILL_MINIMAL_CFG=1
export QWEN36_PREFILL_MM_FP32_ACC=0
export QWEN36_PREFILL_MM_PACKER_L1_ACC=1
export QWEN36_PREFILL_OVERLAP=1
export QWEN36_PREFILL_PROGCFG=1
export QWEN36_PREFILL_PROGCFG_OVERRIDES=1
export QWEN36_PREFIX_WRITE_ITERS=100
export QWEN36_PREFIX_WRITE_WIDTH=1
export QWEN36_ROPE_DEVICE_TABLE=1
export QWEN36_ROPE_LEGACY=0

export QWEN9B_MLP_DOWN_AUTO=0
export QWEN9B_MLP_UP_AUTO=0
export QWEN9B_SDPA_QK64=0

export QWEN_BATCHED_GROUPED=1
export QWEN_GDN_DIAG_ALPHA=0.25
export QWEN_GDN_FP32_STATE=0
export QWEN_GDN_INV_DOUBLING=0
export QWEN_SDPA_BF8=0

# Deliberately LEFT UNSET (see the comment above): bare-truthy or dynamic-default flags.
#   QWEN35_NO_THINK                 - bare truthy: any value seeds an empty <think> block.
#   QWEN35_REF_PROMPT               - bare truthy: any value switches to the 64k reference prompt.
#   QWEN36_CAPACITY_TEST_LAYER_INDEX / QWEN36_CAPACITY_TEST_N_LAYERS - bare truthy TEST-harness only.
#   QWEN36_GDN_CONV_CHUNKS          - bare truthy override; unset = auto chunk count.
#   QWEN36_GDN_CONV_KDA_CCS         - fallback is str(channel_chunk_size), a runtime local.
#   QWEN36_GDN_CONV_XIN_L1_MAX_T    - fallback is str(xin_l1_max_t), a runtime local.
#   QWEN36_GDN_FLA_INPUTS_DRAM      - fallback depends on QWEN_GDN_PATH ("1" if fused else "0");
#                                     leaving it unset lets the code compute the right value for
#                                     whichever of f12/f13 this invocation is running.
#   QWEN36_MAX_TOKENS_ALL_USERS     - bare truthy; vLLM-serving only, unused by this bench.
#   QWEN36_SDPA_PREFILL_CHUNKS      - bare truthy; experimental gated-attention path only.
#   QWEN9B_GDN_DBG                  - bare truthy debug-print switch.
#   QWEN_GDN_PATH                   - set explicitly below per CONFIG (f12: empty: f13: "fused").
export QWEN_GDN_PATH=""

FUSED_FLA_GREP() {
  grep -q "QWEN_GDN_PATH" \
    "$REPO_ROOT/ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule.cpp" \
    2>/dev/null \
  || grep -q "QWEN_GDN_PATH" \
    "$REPO_ROOT/ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/compute/chunk_gated_delta_rule.cpp" \
    2>/dev/null
}

if [ "$CONFIG" = "f13" ]; then
  if ! FUSED_FLA_GREP; then
    echo "ERROR: --config f13 (QWEN_GDN_PATH=fused) requested, but this tree's" >&2
    echo "       chunk_gated_delta_rule.cpp does not mention QWEN_GDN_PATH -- the fused FLA prim" >&2
    echo "       is not wired into this build. Build/checkout the fla-fused-eval branch" >&2
    echo "       (or whichever tree has the fused prim) and re-run there. Refusing to run." >&2
    exit 1
  fi
  export QWEN_GDN_PATH="fused"
  export QWEN_GDN_NP=6
  echo "== CONFIG=f13: QWEN_GDN_PATH=fused QWEN_GDN_NP=6 =="
else
  echo "== CONFIG=f12: phased chunk-parallel GDN prefill (QWEN_GDN_PATH unset) =="
fi

# ---------------------------------------------------------------------------------------------
# 4. Device must be free: exactly one process on the device at a time.
# ---------------------------------------------------------------------------------------------
if [ -e /dev/tenstorrent/0 ]; then
  HOLDERS="$(lsof /dev/tenstorrent/0 2>/dev/null || true)"
  if [ -n "$HOLDERS" ]; then
    echo "ERROR: /dev/tenstorrent/0 is already held by another process:" >&2
    echo "$HOLDERS" >&2
    echo "Refusing to start -- only one process may hold the device at a time." >&2
    exit 1
  fi
else
  echo "WARNING: /dev/tenstorrent/0 not found -- skipping the busy-device check." >&2
fi

# ---------------------------------------------------------------------------------------------
# 5. tt-smi summary (best-effort; bench_e2e_p150.py also captures this into the JSON header).
# ---------------------------------------------------------------------------------------------
if command -v tt-smi >/dev/null 2>&1; then
  tt-smi -s 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    dev0 = (data.get('device_info') or [{}])[0]
    board = (dev0.get('board_info') or {}).get('board_type', 'N/A')
    fw = (dev0.get('firmwares') or {}).get('arc_fw', 'N/A')
    aiclk = (dev0.get('telemetry') or {}).get('aiclk', 'N/A')
    print(f'tt-smi: board_type={board} arc_fw={fw} aiclk={aiclk}')
except Exception as e:
    print(f'tt-smi: could not parse -s output ({e})')
" || echo "tt-smi: -s query failed"
else
  echo "tt-smi: not found on PATH -- skipping"
fi

# ---------------------------------------------------------------------------------------------
# 6. Run.
# ---------------------------------------------------------------------------------------------
RESULTS_DIR="$REPO_ROOT/models/demos/blackhole/qwen36/demo/bench_results"
mkdir -p "$RESULTS_DIR"
if [ ! -f "$RESULTS_DIR/.gitignore" ]; then
  {
    echo "*"
    echo "!.gitignore"
    echo "!README.md"
  } > "$RESULTS_DIR/.gitignore"
fi

if [ "$ISL" = "demo" ]; then
  OUT="$RESULTS_DIR/${CONFIG}_demo_osl${OSL}_$(date +%Y%m%d_%H%M%S).json"
  PROMPT_ARGS=(--demo-prompt)
else
  OUT="$RESULTS_DIR/${CONFIG}_isl${ISL}_osl${OSL}_$(date +%Y%m%d_%H%M%S).json"
  PROMPT_ARGS=(--isl "$ISL")
fi
echo "== running: isl=$ISL osl=$OSL runs=$RUNS chunk=2048 -> $OUT =="

set +e
timeout 1800 python3 "$SCRIPT_DIR/bench_e2e_p150.py" \
  "${PROMPT_ARGS[@]}" --osl "$OSL" --runs "$RUNS" --out "$OUT"
STATUS=$?
set -e

echo "== exit status: $STATUS =="
exit "$STATUS"
