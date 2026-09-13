#!/usr/bin/env bash
# LTX-2.5 distilled 1080p DiffVAE pipeline run. Extra pytest flags pass straight through:
#   bash models/tt_dit/experimental/scripts/run_ltx25_pipeline.sh
# Deep profile, outputs kept per run (see the PROFILE block at the bottom):
#   PROFILE=1 bash models/tt_dit/experimental/scripts/run_ltx25_pipeline.sh
#
# Exists because pasting the env prefix as one quoted multi-line command keeps corrupting it:
# once via non-breaking spaces glued to the first var on each line, once via raw newlines
# splitting it into five bash statements so pytest ran with no arguments.
set -euo pipefail

# Repo root, four levels up from this script, so the checkout can live anywhere.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

export TT_DIT_CACHE_DIR="${TT_DIT_CACHE_DIR:-$HOME/.cache/tt-dit}"
# Overridable so a checkout whose weights live elsewhere (e.g. the HF cache) can point at
# them without editing this file -- the sibling run_ltx25_diffvae.sh already works this way.
export LTX25_ROOT=${LTX25_ROOT:-/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941}
export LTX25_DIFFVAE=${LTX25_DIFFVAE:-1}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

export NO_PROMPT=${NO_PROMPT:-1}
export RUN_WARMUP=${RUN_WARMUP:-1}
export LTX_TRACED=${LTX_TRACED:-0}
export SEED=${SEED:-10}

#export LTX_YUV_EXPORT=1

export NUM_FRAMES=${NUM_FRAMES:-145}
export HEIGHT=${HEIGHT:-1088}
export WIDTH=${WIDTH:-1920}
export OUTPUT_PATH="${OUTPUT_PATH:-$HOME/ltx25_diffvae_1080p.mp4}"

export DIFFVAE_SLAB_FRAMES=${DIFFVAE_SLAB_FRAMES:-78}
export DIFFVAE_STAGES_WSP=1
export DIFFVAE_DEVICE_NOISE=1
export DIFFVAE_DEVICE_PREPROC=1
export DIFFVAE_DEVICE_UNPATCHIFY=1
export DIFFVAE_TRIM_PAD_CHANNELS=1
export DIFFVAE_DET_COLPAR_QKV=1
export DIFFVAE_DET_FUSED_ROPE=1
export DIFFVAE_DET_FUSED_SWIGLU=1
export DIFFVAE_STAGE_TIMING=1
# Stream one "[stage HH:MM:SS] > label" / "< label  N ms" line per decode span so a hang is visible
# while it happens (the last ">" with no "<" names it) instead of at the timeout. Tree unchanged.
export DIFFVAE_STAGE_LOG=${DIFFVAE_STAGE_LOG:-1}
# Stage 5 and the W-sharded deterministic stages 1-3 run the bricked executor
# (bricked_sp_w_sharded). The only other value either knob accepts is "linear_order" (replicated),
# which does not fit the pipeline's memory.
export DIFFVAE_STAGE5_BACKEND=${DIFFVAE_STAGE5_BACKEND:-bricked_sp_w_sharded}
export DIFFVAE_STAGES_BACKEND=${DIFFVAE_STAGES_BACKEND:-bricked_sp_w_sharded}
export DIFFVAE_S5_GNA_STRIDE=${DIFFVAE_S5_GNA_STRIDE:-1,1,1}
export DIFFVAE_TP_HEADS=${DIFFVAE_TP_HEADS:-1}

# Fail loudly if the paste-corruption bugs ever come back rather than running a half-configured job.
: "${LTX25_DIFFVAE:?}" "${LTX25_ROOT:?}"
[ -f "$LTX25_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors" ] || {
  echo "DiffVAE weights not under LTX25_ROOT=$LTX25_ROOT" >&2; exit 1; }

PYTEST=(python_env/bin/python -u -m pytest
  models/tt_dit/tests/models/ltx/test_pipeline_ltx25_distilled.py::test_pipeline_ltx25_distilled
  -k 4x8sp1tp0nl2_ring_is_fsdp0 -s -q --timeout=0)

# PROFILE=1: one untraced gen with the deep decode-tree profile, everything under a fresh
# generated/profile/<UTC stamp>/ (mp4, log.txt, decode_trees.txt) so reruns never overwrite.
# The PERFORMANCE table's VAE decode row expands into stage/block sub-rows (LTX_PERF_BREAKDOWN).
# Untraced because the tree spans time trace capture, not execution, and BLOCK_PROF's syncs
# cannot live inside a trace -- so totals here are slower than traced production numbers.
# The tree is printed by decode_tree_plugin.py (the vae conftest fixture does not reach this test).
if [ "${PROFILE:-0}" = 1 ]; then
  OUT_DIR="$PWD/generated/profile/$(date -u +%Y%m%d_%H%M%S)"
  mkdir -p "$OUT_DIR"
  export LTX_TRACED=0
  export LTX_VAE_TIME=${LTX_VAE_TIME:-1}
  export DIFFVAE_BLOCK_PROF=${DIFFVAE_BLOCK_PROF:-1}
  export DIFFVAE_SLAB_FRAMES=73
  export OUTPUT_PATH="$OUT_DIR/ltx25_1080p.mp4"
  export DIFFVAE_TREE_OUT="$OUT_DIR/decode_trees.txt"
  # Expand the perf table's "VAE decode" row from the decode tree (levels deep), and with BLOCK_PROF
  # add the exclusive-by-category block. 0 keeps the flat table.
  export LTX_PERF_BREAKDOWN=${LTX_PERF_BREAKDOWN:-2}
  export PYTHONPATH="$PWD/models/tt_dit/experimental/scripts${PYTHONPATH:+:$PYTHONPATH}"
  echo "[profile] $(git rev-parse --short HEAD) -> $OUT_DIR"
  set +e
  "${PYTEST[@]}" -p decode_tree_plugin "$@" 2>&1 | tee "$OUT_DIR/log.txt"
  status=${PIPESTATUS[0]}
  echo "PYTEST_EXIT=$status" | tee -a "$OUT_DIR/log.txt"
  exit "$status"
fi

exec "${PYTEST[@]}" "$@"
