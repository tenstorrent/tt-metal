#!/usr/bin/env bash
# LTX-2.5 distilled 1080p DiffVAE pipeline run. Extra pytest flags pass straight through:
#   bash models/tt_dit/experimental/scripts/run_ltx25_pipeline.sh --timeout=0
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

export DIFFVAE_GNA=1
export DIFFVAE_SLAB_FRAMES=${DIFFVAE_SLAB_FRAMES:-78}
export DIFFVAE_BLOCK=1
export DIFFVAE_SP_FUSED=1
export DIFFVAE_STAGES_WSP=1
export DIFFVAE_SDPA_KCHUNK=256
export DIFFVAE_PAD_GATHER=1
export DIFFVAE_DEVICE_NOISE=1
export DIFFVAE_DEVICE_PREPROC=1
export DIFFVAE_DEVICE_UNPATCHIFY=1
export DIFFVAE_TRIM_PAD_CHANNELS=1
export DIFFVAE_DET_COLPAR_QKV=1
export DIFFVAE_DET_FUSED_ROPE=1
export DIFFVAE_DET_FUSED_SWIGLU=1
export DIFFVAE_DET_FLAT_SEQ=1
export DIFFVAE_S5_FLAT_SEQ=1
export DIFFVAE_STAGE_TIMING=1

# Fail loudly if the paste-corruption bugs ever come back rather than running a half-configured job.
: "${LTX25_DIFFVAE:?}" "${LTX25_ROOT:?}"
[ -f "$LTX25_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors" ] || {
  echo "DiffVAE weights not under LTX25_ROOT=$LTX25_ROOT" >&2; exit 1; }

exec python_env/bin/python -u -m pytest \
  models/tt_dit/tests/models/ltx/test_pipeline_ltx25_distilled.py::test_pipeline_ltx25_distilled \
  -k 4x8sp1tp0nl2_ring_is_fsdp0 -s -q "$@"
