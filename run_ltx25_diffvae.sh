#!/usr/bin/env bash
# LTX-2.5 distilled 1080p DiffVAE decode, 145 frames at 1920x1088 on a 4x8 mesh.
#
# Exists because pasting the env prefix as one quoted multi-line command keeps corrupting it:
# once via non-breaking spaces glued to the first var on each line (job 770/752), once via raw
# newlines splitting it into five bash statements so pytest ran with no arguments (job 790).
#
# ── THE BASELINE ──────────────────────────────────────────────────────────────────────────────
# Every timing comparison in the decode-tree work is against this exact invocation (job 864):
#
#   tt-device-mcp run -t 10000 "DIFFVAE_BLOCK_PROF=1 bash ./run_ltx25_diffvae.sh --timeout=0"
#
#   -> decode TOTAL 10706.5 ms | det stages 2309.4 (21.6%) | stage 5 8396.0 (78.4%)
#      reshape+permute 2942.4 (27.5%) | allgather 2569.1 (24.0%) | sdpa 887.3 (8.3%)
#
# DIFFVAE_BLOCK_PROF=1 is part of the baseline, not a decoration: without it the deterministic
# stages show only their na3d collectives and the seven attention spans (kv-wrow, q-to-seq, the
# block permutes, out-proj) do not exist -- that run is ~10600 ms and NOT comparable.
# --timeout=0 disables pytest.ini's 300 s per-test limit, which a real 145-frame decode exceeds.
#
# Arms measured against it, same line with one variable prefixed:
#   DIFFVAE_KV_BF8=1       -> 10096.0 ms (-610, -5.7%); all kv-allgather, kv-wrow unmoved
#   DIFFVAE_KV_RM_GATHER=1 -> 10323.9 ms (-383, -3.6%); kv-wrow GONE (-1213), gather +845
#   DIFFVAE_BLOCK=0        -> 21847.5 ms (2.04x SLOWER); also disables GNA, so sdpa explodes 26x
#
# DIFFVAE_KV_RM_PAGE sweep (with RM_GATHER=1) -- page size is NOT why the tiled gather wins:
#   9984 B (default) 10324 | 4992 B 11875 | 1664 B 11931 | 768 B 30639 | 256 B 38983
# Monotonic, and the largest reachable page is already the default. RM at its best page is still
# 39%/block slower than TILE at 2048 B, so the tiled path's edge is elsewhere (tile-granular
# chunking, or hyperparams tuned for it). The full win needs the fused reader to read TILE K/V.
#
# Extra pytest flags pass straight through: bash ./run_ltx25_diffvae.sh --timeout=0 -k s16
set -euo pipefail

cd /home/jameslee/tt-metal

export TT_DIT_CACHE_DIR="${TT_DIT_CACHE_DIR:-$HOME/.cache/tt-dit}"
export LTX25_ROOT=${LTX25_ROOT:-/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941}
export LTX25_DIFFVAE=${LTX25_DIFFVAE:-1}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
# The models/tt_dit/tests/models/vae/* tests do NOT read LTX25_ROOT -- they resolve weights from
# DIFFVAE_CHECKPOINT, defaulting to ~/.cache/ltx-checkpoints/ltx-2.5/... which does not exist here,
# and skip silently when it is missing.
export DIFFVAE_CHECKPOINT="${DIFFVAE_CHECKPOINT:-$LTX25_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors}"

# Only read by test_decode_timing.py: collectives default to Linear and 1 link, unlike the pipeline
# runs, which used ring + 2 links. Uncomment to match those.
# export DIFFVAE_TOPOLOGY=ring
# export DIFFVAE_NUM_LINKS=2
# export DIFFVAE_LATENT_T=19   # default; 19 latent frames -> 145 output frames

export NO_PROMPT=${NO_PROMPT:-1}
export RUN_WARMUP=${RUN_WARMUP:-0}
export LTX_TRACED=${LTX_TRACED:-0}
export SEED=${SEED:-10}

export NUM_FRAMES=${NUM_FRAMES:-145}
export HEIGHT=${HEIGHT:-1088}
export WIDTH=${WIDTH:-1920}
export OUTPUT_PATH="${OUTPUT_PATH:-$HOME/ltx25_diffvae_1080p.mp4}"

export DIFFVAE_GNA=${DIFFVAE_GNA:-1}
# 73 OOMs on the W-SP path: band 0 spans 78 frames -> the full-W K/V gather is 651.8 MB/bank against
# a 582.2 MB largest free block. 48 -> band 53 -> 442.9 MB/bank. (~8.36 MB/bank per band frame.)
export DIFFVAE_SLAB_FRAMES=${DIFFVAE_SLAB_FRAMES:-73} # changing to 48 fixes the issue!!
export DIFFVAE_BLOCK=${DIFFVAE_BLOCK:-1}
export DIFFVAE_SP_FUSED=${DIFFVAE_SP_FUSED:-1}
export DIFFVAE_STAGES_WSP=${DIFFVAE_STAGES_WSP:-1}
export DIFFVAE_SDPA_KCHUNK=${DIFFVAE_SDPA_KCHUNK:-256}
export DIFFVAE_PAD_GATHER=${DIFFVAE_PAD_GATHER:-1}
export DIFFVAE_DEVICE_NOISE=${DIFFVAE_DEVICE_NOISE:-1}
export DIFFVAE_DEVICE_PREPROC=${DIFFVAE_DEVICE_PREPROC:-1}
export DIFFVAE_DEVICE_UNPATCHIFY=${DIFFVAE_DEVICE_UNPATCHIFY:-1}
export DIFFVAE_TRIM_PAD_CHANNELS=${DIFFVAE_TRIM_PAD_CHANNELS:-1}
export DIFFVAE_DET_COLPAR_QKV=${DIFFVAE_DET_COLPAR_QKV:-1}
export DIFFVAE_DET_FUSED_ROPE=${DIFFVAE_DET_FUSED_ROPE:-1}
export DIFFVAE_DET_FUSED_SWIGLU=${DIFFVAE_DET_FUSED_SWIGLU:-1}
export DIFFVAE_DET_FLAT_SEQ=${DIFFVAE_DET_FLAT_SEQ:-1}
export DIFFVAE_S5_FLAT_SEQ=${DIFFVAE_S5_FLAT_SEQ:-1}
export DIFFVAE_STAGE_TIMING=${DIFFVAE_STAGE_TIMING:-1}
# Tree + per-stage/category breakdown, rendered at test teardown; DIFFVAE_TREE_ALL=1 also renders
# the warm-up pass. Prefix DIFFVAE_BLOCK_PROF=1 to break the deterministic stages down into
# attention/mlp as well -- that adds 64 device syncs, measured at +0.4% on det stages and +0.15% on
# the decode (job 845 vs 846), so the two modes are close but should not be mixed in one comparison.
export DIFFVAE_TP_HEADS=${DIFFVAE_TP_HEADS:-1}

# Fail loudly if the paste-corruption bugs ever come back rather than running a half-configured job.
: "${LTX25_DIFFVAE:?}" "${LTX25_ROOT:?}"
[ -f "$DIFFVAE_CHECKPOINT" ] || {
  echo "DiffVAE weights missing: DIFFVAE_CHECKPOINT=$DIFFVAE_CHECKPOINT" >&2; exit 1; }

exec python_env/bin/python -u -m pytest \
	models/tt_dit/tests/models/vae/test_decode_timing.py::test_decode_wsp_timing -k s34x60 -x -q -s "$@"
