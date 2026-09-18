#!/usr/bin/env bash
# LTX-2.5 distilled 1080p DiffVAE pipeline run. Extra pytest flags pass straight through:
#   bash models/tt_dit/experimental/scripts/run_ltx25_pipeline.sh
# Deep profile, outputs kept per run (see the PROFILE block at the bottom):
#   PROFILE=1 bash models/tt_dit/experimental/scripts/run_ltx25_pipeline.sh
#
# How the DiffVAE is built is a pytest option, not an environment variable: --diffvae selects it and
# the --diffvae-* options (see `pytest --help`, group "LTX-2.5 DiffVAE") say how it runs, defaulting
# to DiffVAEOptions.production(). This script passes only --diffvae-slab-frames; add anything else
# after the script name.
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

# Stage-5 frames per band: the one DiffVAE knob this script varies (see PROFILE and LTX_TRACED below).
SLAB_FRAMES=${SLAB_FRAMES:-78}
# The decode tree. Off under LTX_TRACED=1: with the DiffVAE traced its spans would synchronise the
# mesh inside the capture, and the decoder refuses to capture with the timers on.
if [ "$LTX_TRACED" = 1 ]; then
  export TT_DIT_STAGE_TIMING=${TT_DIT_STAGE_TIMING:-0}
else
  export TT_DIT_STAGE_TIMING=${TT_DIT_STAGE_TIMING:-1}
fi
# Stream one "[stage HH:MM:SS] > label" / "< label  N ms" line per decode span so a hang is visible
# while it happens (the last ">" with no "<" names it) instead of at the timeout. Tree unchanged.
export TT_DIT_STAGE_LOG=${TT_DIT_STAGE_LOG:-1}

# Fail loudly if the paste-corruption bugs ever come back rather than running a half-configured job.
: "${LTX25_ROOT:?}"
# Under LTX_TRACED=1 the transformer's captures and the 1.9 GiB a decode leaves resident put the
# stage-5 MLP hidden (2.4 GB at slab 73) past the largest contiguous DRAM block; 48 is the measured
# fit (2026-09-15). Refuse here, at t=0, instead of in gen #0's decode after two minutes of loading.
if [ "$LTX_TRACED" = 1 ] && [ "$SLAB_FRAMES" -gt 48 ]; then
  echo "LTX_TRACED=1 needs SLAB_FRAMES<=48 (got $SLAB_FRAMES): larger slabs OOM in the traced decode" >&2
  exit 1
fi
[ -f "$LTX25_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors" ] || {
  echo "DiffVAE weights not under LTX25_ROOT=$LTX25_ROOT" >&2; exit 1; }

PYTEST=(python_env/bin/python -u -m pytest
  models/tt_dit/tests/models/ltx/test_pipeline_ltx25_distilled.py::test_pipeline_ltx25_distilled
  -k 4x8sp1tp0nl2_ring_is_fsdp0 -s -q --timeout=0 --diffvae)

# PROFILE=1: one untraced gen with the deep decode-tree profile, everything under a fresh
# generated/profile/<UTC stamp>/ (mp4, log.txt, decode_trees.txt) so reruns never overwrite.
# The PERFORMANCE table's VAE decode row expands into stage/block sub-rows (LTX_PERF_BREAKDOWN).
# Untraced because the tree spans time trace capture, not execution, and BLOCK_PROF's syncs
# cannot live inside a trace -- so totals here are slower than traced production numbers.
# The tree is printed by timing_tree_plugin.py (this test does not request the timing_tree fixture).
if [ "${PROFILE:-0}" = 1 ]; then
  OUT_DIR="$PWD/generated/profile/$(date -u +%Y%m%d_%H%M%S)"
  mkdir -p "$OUT_DIR"
  export LTX_TRACED=0
  export LTX_VAE_TIME=${LTX_VAE_TIME:-1}
  export TT_DIT_BLOCK_PROF=${TT_DIT_BLOCK_PROF:-1}
  SLAB_FRAMES=73
  export OUTPUT_PATH="$OUT_DIR/ltx25_1080p.mp4"
  export TT_DIT_TREE_OUT="$OUT_DIR/decode_trees.txt"
  # Expand the perf table's "VAE decode" row from the decode tree (levels deep), and with BLOCK_PROF
  # add the exclusive-by-category block. 0 keeps the flat table.
  export LTX_PERF_BREAKDOWN=${LTX_PERF_BREAKDOWN:-2}
  export PYTHONPATH="$PWD/models/tt_dit/experimental/scripts${PYTHONPATH:+:$PYTHONPATH}"
  echo "[profile] $(git rev-parse --short HEAD) -> $OUT_DIR"
  set +e
  "${PYTEST[@]}" --diffvae-slab-frames "$SLAB_FRAMES" -p timing_tree_plugin "$@" 2>&1 | tee "$OUT_DIR/log.txt"
  status=${PIPESTATUS[0]}
  echo "PYTEST_EXIT=$status" | tee -a "$OUT_DIR/log.txt"
  exit "$status"
fi

exec "${PYTEST[@]}" --diffvae-slab-frames "$SLAB_FRAMES" "$@"
