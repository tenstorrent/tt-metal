#!/usr/bin/env bash
# LTX-2.5 distilled 1080p DiffVAE run. Extra pytest flags can be appended: bash run_ltx25_diffvae.sh --timeout=0
#
# Exists because pasting the env prefix as one quoted multi-line command keeps corrupting it:
# once via non-breaking spaces glued to the first var on each line (job 770/752), once via raw
# newlines splitting it into five bash statements so pytest ran with no arguments (job 790).
set -euo pipefail

cd /home/jameslee/tt-metal

export TT_DIT_CACHE_DIR="$HOME/.cache/tt-dit"
export LTX25_ROOT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941
export LTX25_DIFFVAE=1
export HF_HUB_DISABLE_XET=1
# The models/tt_dit/tests/models/vae/* tests do NOT read LTX25_ROOT -- they resolve weights from
# DIFFVAE_CHECKPOINT, defaulting to ~/.cache/ltx-checkpoints/ltx-2.5/... which does not exist here,
# and skip silently when it is missing.
export DIFFVAE_CHECKPOINT="$LTX25_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors"

# Only read by test_decode_timing.py: collectives default to Linear and 1 link, unlike the pipeline
# runs, which used ring + 2 links. Uncomment to match those.
# export DIFFVAE_TOPOLOGY=ring
# export DIFFVAE_NUM_LINKS=2
# export DIFFVAE_LATENT_T=19   # default; 19 latent frames -> 145 output frames

export NO_PROMPT=1
export RUN_WARMUP=0
export LTX_TRACED=0
export SEED=10

export NUM_FRAMES=145
export HEIGHT=1088
export WIDTH=1920
export OUTPUT_PATH="$HOME/ltx25_diffvae_1080p.mp4"

export DIFFVAE_GNA=1
# 73 OOMs on the W-SP path: band 0 spans 78 frames -> the full-W K/V gather is 651.8 MB/bank against
# a 582.2 MB largest free block. 48 -> band 53 -> 442.9 MB/bank. (~8.36 MB/bank per band frame.)
export DIFFVAE_SLAB_FRAMES=73 # changing to 48 fixes the issue!!
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
# Tree + per-stage/category breakdown, rendered at test teardown; DIFFVAE_TREE_ALL=1 also renders
# the warm-up pass. Prefix DIFFVAE_BLOCK_PROF=1 to break the deterministic stages down into
# attention/mlp as well -- that adds 64 device syncs, measured at +0.4% on det stages and +0.15% on
# the decode (job 845 vs 846), so the two modes are close but should not be mixed in one comparison.
export DIFFVAE_TP_HEADS=1

# Fail loudly if the paste-corruption bugs ever come back rather than running a half-configured job.
: "${LTX25_DIFFVAE:?}" "${LTX25_ROOT:?}"
[ -f "$DIFFVAE_CHECKPOINT" ] || {
  echo "DiffVAE weights missing: DIFFVAE_CHECKPOINT=$DIFFVAE_CHECKPOINT" >&2; exit 1; }

exec python_env/bin/python -u -m pytest \
	models/tt_dit/tests/models/vae/test_decode_timing.py::test_decode_wsp_timing -k s34x60 -x -q -s "$@"
