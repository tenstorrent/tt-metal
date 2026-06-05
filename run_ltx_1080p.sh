#!/usr/bin/env bash
# Generate a 1080p LTX video (Fast or Pro) on Wormhole-LB for ANY prompt.
#
# Why two phases?  Running the ~6 GB Gemma text encoder inline fragments DRAM enough that
# the full-res (1920x1088) VAE decode can't find its large contiguous block on WH-LB. So we
# (1) pre-encode the prompt in a throwaway process — this caches the prompt embedding to host
# (~/.cache/tt-dit/ltx-embeddings) — then (2) run the real generate in a fresh process where
# the cached embedding skips the encoder entirely and the decode runs from a clean heap.
# This is fully quality-neutral (identical numerics to a single clean run).
#
# Usage:
#   ./run_ltx_1080p.sh [fast|pro] "<prompt>" [output.mp4]
#
# Env overrides: HEIGHT (1088) WIDTH (1920) NUM_FRAMES (145) SEED (10) MESH_K (2x4sp0tp1)
set -euo pipefail
cd "$(dirname "$0")"

# Wormhole-only entry point — BH workflows use the pytest commands in models/tt_dit/models/LTX2.md.
export TT_METAL_HOME="$(pwd)"
export PYTHONPATH="${PYTHONPATH:-$HOME/.local/lib/python3.10/site-packages}:$(pwd):$(pwd)/ttnn:$(pwd)/tools"
if ! python3 -c "import ttnn; raise SystemExit(0 if ttnn.device.is_wormhole_b0() else 1)" 2>/dev/null; then
  echo "run_ltx_1080p.sh is for Wormhole (wormhole_b0) only. On Blackhole, use LTX2.md pytest examples." >&2
  exit 1
fi

PIPELINE="${1:-fast}"
PROMPT_ARG="${2:-}"
OUT_ARG="${3:-}"

case "$PIPELINE" in
  fast) TEST="models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled_av.py" ;;
  pro)  TEST="models/tt_dit/tests/models/ltx/test_pipeline_ltx.py" ;;
  *) echo "usage: $0 [fast|pro] \"<prompt>\" [output.mp4]" >&2; exit 2 ;;
esac

export TT_DIT_CACHE_DIR="${TT_DIT_CACHE_DIR:-$HOME/.cache/tt-dit}"

# --- Run config ---
export NO_PROMPT=1
export HEIGHT="${HEIGHT:-1088}"     # 1080p target (1088 is the nearest /64-divisible height)
export WIDTH="${WIDTH:-1920}"
export NUM_FRAMES="${NUM_FRAMES:-145}"
export SEED="${SEED:-10}"
MESH_K="${MESH_K:-2x4sp0tp1}"       # WH 2x4 Loud Box; use 2x2sp0tp1 / wh_4x8sp1tp0 for other configs
if [[ -n "$PROMPT_ARG" ]]; then export PROMPT="$PROMPT_ARG"; fi
if [[ -n "$OUT_ARG" ]]; then export OUTPUT_PATH="$OUT_ARG"; fi

echo "=== Phase 1/2: pre-encoding prompt (caching embedding to host) ==="
# Encode-only: caches the embedding then returns before denoise/VAE. Tolerate a nonzero exit
# (the embedding is written during encode regardless) so a quirk here can't block Phase 2.
LTX_ENCODE_ONLY=1 pytest "$TEST" -k "$MESH_K" -s --timeout 1800 || true

echo "=== Phase 2/2: generating 1080p video (cached embedding -> clean decode) ==="
pytest "$TEST" -k "$MESH_K" -s --timeout 5400
