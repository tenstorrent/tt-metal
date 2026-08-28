#!/usr/bin/env bash
# LTX-2.5 distilled 1080p DiffVAE decode, 145 frames at 1920x1088 on a 4x8 mesh.
#
# Exists because pasting the env prefix as one quoted multi-line command keeps corrupting it:
# once via non-breaking spaces glued to the first var on each line, once via raw newlines
# splitting it into five bash statements so pytest ran with no arguments.
#
# ── THE BASELINE ──────────────────────────────────────────────────────────────────
# Every timing comparison in the decode-tree work is against this exact invocation:
#
#   tt-device-mcp run -t 1500 \
#     "DIFFVAE_BLOCK_PROF=1 bash models/tt_dit/experimental/scripts/run_ltx25_diffvae.sh --timeout=0"
#
#   -> decode TOTAL 8817.6 ms | det stages 1782.0 (20.2%) | stage 5 7033.1 (79.8%)
#
# HOW IT GOT HERE -- two variables, both now defaults below, measured one at a time on this
# same invocation. The deltas are disjoint (different rows of the tree) so they sum exactly:
#
#   10752.8 ms   Topology.Linear + num_links=1, DET_FUSED_QKV off   (the old baseline)
#   -1710.9      DIFFVAE_TOPOLOGY=ring + DIFFVAE_NUM_LINKS=2        (-15.9%)
#    -498.6      DIFFVAE_DET_FUSED_QKV=1                            (-5.5%)
#   ---------
#    8543.3 ms   -2209.5 total (-20.5%)
#
# All four of those were measured with the deep-profile spans ABSENT. Those spans (qkv-proj,
# qkv-norm, qkv-rope, norm+modulate, residual crop+add, na3d gather) landed afterwards and cost
# ~275 ms of syncs, which is the whole gap between 8543.3 and the 8817.6 above. They exist only
# under BLOCK_PROF=1 -- which the baseline sets -- so 8543.3 is not reachable on this tree.
# Compare new arms against 8817.6.
#
#   ring + 2 links     kv-allgather 2162.8 -> 840.1, head-allgather 409.3 -> 129.9. The fabric is
#                      built FABRIC_1D_RING either way, so Linear left the wraparound link idle and
#                      num_links=1 used one of the two eth channels crossing the size-8 axis. Every
#                      non-collective row moved by <1.5 ms. num_links is capped at 2 by the fabric.
#   det fused qkv      det stage 0 947.1 -> 533.2. It is the one stage built with tp_axis=None, so
#                      colpar_qkv is False and DET_COLPAR_QKV/DET_FUSED_ROPE never reached it: three
#                      2048->2048 GEMMs and the matmul-based apply_rope. Fusing gives it the same
#                      path stages 1-3 already run -- rope 335.9 -> 18.1, proj 214.8 -> 40.2, against
#                      +77.1 on qkv-to-volume for the permute nlp_create_qkv_heads' layout forces.
#                      Gated: test_det_nablock_arms.py stage1 arms -- bit-exact (100.0000%) for the
#                      projection fusion, 99.9985% once the fused rope comes with it.
#
# PCC, both changes together, vs the 8 committed diffvae_gate baselines (run_diffvae_gates.sh with
# DIFFVAE_DET_FUSED_QKV=1 DIFFVAE_TOPOLOGY=ring DIFFVAE_NUM_LINKS=2): every gate within 0.0001
# percentage points, against a 0.02-point tolerance. Six exact, decoder context -0.0001, end-to-end
# decode +0.0001 (99.8817 -> 99.8818). The ring/2-link run and a Linear/1-link run of the same
# suite gave IDENTICAL PCCs to four decimals, which is the direct evidence that the collective
# change is numerically inert -- an all-gather only moves bytes. So the whole -2209.5 ms is free.
#
# The gates pinned Linear/1-link at all three CCLManager sites and could not see the collective
# config the runner ships; they now go through _gate_ccl(), which reads DIFFVAE_TOPOLOGY /
# DIFFVAE_NUM_LINKS and defaults to Linear/1-link so an unset environment still reproduces the
# committed baseline exactly.
#
# DIFFVAE_BLOCK_PROF=1 is part of the baseline, not a decoration: without it the deterministic
# stages show only their na3d collectives and the attention spans (kv-wrow, q-to-seq, the block
# permutes, qkv-proj, out-proj) do not exist -- that run is NOT comparable.
# --timeout=0 disables pytest.ini's 300 s per-test limit, which a real 145-frame decode exceeds.
#
# Extra pytest flags pass straight through:
#   bash models/tt_dit/experimental/scripts/run_ltx25_diffvae.sh --timeout=0 -k s16
set -euo pipefail

# Repo root, four levels up from this script, so the checkout can live anywhere.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

export TT_DIT_CACHE_DIR="${TT_DIT_CACHE_DIR:-$HOME/.cache/tt-dit}"
export LTX25_ROOT=${LTX25_ROOT:-/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941}
export LTX25_DIFFVAE=${LTX25_DIFFVAE:-1}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
# The models/tt_dit/tests/models/vae/* tests do NOT read LTX25_ROOT -- they resolve weights from
# DIFFVAE_CHECKPOINT, defaulting to ~/.cache/ltx-checkpoints/ltx-2.5/... which does not exist here,
# and skip silently when it is missing.
export DIFFVAE_CHECKPOINT="${DIFFVAE_CHECKPOINT:-$LTX25_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors}"

# Only read by test_decode_timing.py. Ring + 2 links, matching the pipeline runs: the fabric is
# built as FABRIC_1D_RING either way, so Linear left the wraparound link enabled and unused, and
# num_links=1 used one of the two eth channels that reach across the size-8 axis.
export DIFFVAE_TOPOLOGY=${DIFFVAE_TOPOLOGY:-ring}
export DIFFVAE_NUM_LINKS=${DIFFVAE_NUM_LINKS:-2}
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
export DIFFVAE_SLAB_FRAMES=${DIFFVAE_SLAB_FRAMES:-78} # changing to 48 fixes the issue!!
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
# Reaches ONLY det stage 0: stages 1+ get the fused qkv via COLPAR_QKV already, so this is a no-op
# for them (asserted in test_det_nablock_arms.py). See "HOW IT GOT HERE" above.
export DIFFVAE_DET_FUSED_QKV=${DIFFVAE_DET_FUSED_QKV:-1}
export DIFFVAE_DET_FUSED_ROPE=${DIFFVAE_DET_FUSED_ROPE:-1}
export DIFFVAE_DET_FUSED_SWIGLU=${DIFFVAE_DET_FUSED_SWIGLU:-1}
export DIFFVAE_DET_FLAT_SEQ=${DIFFVAE_DET_FLAT_SEQ:-1}
export DIFFVAE_STAGE_TIMING=${DIFFVAE_STAGE_TIMING:-1}
# Tree + per-stage/category breakdown, rendered at test teardown; DIFFVAE_TREE_ALL=1 also renders
# the warm-up pass. Prefix DIFFVAE_BLOCK_PROF=1 to break the deterministic stages down into
# attention/mlp as well -- that adds 64 device syncs, so the two modes are close but should not be
# mixed in one comparison.
export DIFFVAE_TP_HEADS=${DIFFVAE_TP_HEADS:-1}

# Fail loudly if the paste-corruption bugs ever come back rather than running a half-configured job.
: "${LTX25_DIFFVAE:?}" "${LTX25_ROOT:?}"
[ -f "$DIFFVAE_CHECKPOINT" ] || {
  echo "DiffVAE weights missing: DIFFVAE_CHECKPOINT=$DIFFVAE_CHECKPOINT" >&2; exit 1; }

exec python_env/bin/python -u -m pytest \
	models/tt_dit/tests/models/vae/test_decode_timing.py::test_decode_wsp_timing -k s34x60 -x -q -s "$@"
