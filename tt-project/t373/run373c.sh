#!/bin/bash
# t373: the repo's standard LTX e2e test, unmodified, on ttp/t373 bf45db27371 (t48 90ed8257bac + timing spans + yuv assembly on the export worker by default; own build),
# blx01 4x8, LTX_VERSION=2.5 LTX25_DIFFVAE=0 (real 2.5 conv VAE from LTX25_ROOT), defaults otherwise: 8+3, 1088x1920,
# 145 frames, 24 fps, seed 10, bh_4x8sp1tp0_ring, traced. Usage: run373.sh <tag> <timing>
#   timing=1: TT_DIT_STAGE_TIMING=1 LTX_PERF_BREAKDOWN=2 TT_DIT_STAGE_LOG=1 (VAE row split; spans sync, so not a headline).
# Own JIT cache t373/jit, no DiT cache. Broker: -e env373.yaml -t 570.
if [ -z "$INNER" ]; then
  INNER=1 setsid bash "$0" "$@" & PG=$!
  trap 'kill -TERM -- -$PG 2>/dev/null; sleep 5; kill -KILL -- -$PG 2>/dev/null' EXIT
  trap 'exit 143' TERM; trap 'exit 130' INT
  wait $PG; exit $?
fi
set -o pipefail
TAG=${1:?tag}; TIMING=${2:?timing}
F=/var/tmp/fasth3; T=$F/t373; W=$T/b; WANT=bf45db27371; OUT=$T/out_$TAG; M=$F/models/ltx-2.5
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 70 ] || { echo "[t373] df / $use% > 70%"; exit 5; }
gb=$(timeout 120 du -sxBG $F | cut -f1 | tr -dc 0-9); [ "${gb:-999}" -le 150 ] || { echo "[t373] $F ${gb}G > 150G"; exit 5; }
rm -rf $OUT; mkdir -p $OUT $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp HF_HOME=$F/home/.cache/huggingface HF_HUB_OFFLINE=1
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools PYTHONDONTWRITEBYTECODE=1 TT_METAL_CACHE=$T/jit
export LTX_CHECKPOINT=$F/models/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
export GEMMA_PATH=$F/models/gemma-3-12b-it-qat-q4_0-unquantized
export LTX_VERSION=2.5 LTX25_ROOT=$M LTX25_DIFFVAE=0
export RUN_VBENCH=0 RUN_CLIP=0 LTX_OUT_DIR=$OUT
unset TT_DIT_CACHE_DIR LTX25_VIDEO_VAE LTX_FUSE_YUV_OUTPUT LTX_TRACE_YUV_OUTPUT LTX_VAE_FUSE_UNPATCH LTX_AUDIO_OVERLAP \
  LTX_FUSE_GATE_ON_DEVICE LTX_FUSE_NORM_ADALN LTX_QUANT LTX_QUANT_ACTIVATIONS LTX_QUALITY LTX_FAST \
  LTX_S1_SIGMAS LTX_S2_SIGMAS LTX_TRACED LTX_ITER_ENV NO_PROMPT SEED RUN_WARMUP PROMPT OUTPUT_PATH LTX25_TEXT_STACK \
  LTX_E2E_SEEDS LTX_E2E_EXTRA_REPLAYS LTX_FRESH_PROMPTS LTX_E2E_AB_ENV NUM_FRAMES HEIGHT WIDTH FPS DIFFVAE_SLAB_FRAMES \
  TT_DIT_STAGE_TIMING LTX_PERF_BREAKDOWN TT_DIT_STAGE_LOG TT_DIT_BLOCK_PROF TT_DIT_PLANAR_CONCAT_CACHE
[ "$TIMING" = 1 ] && export TT_DIT_STAGE_TIMING=1 LTX_PERF_BREAKDOWN=2 TT_DIT_STAGE_LOG=1
cd $OUT || exit 3
TF=$W/models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py
HEAD=$(git -C $W rev-parse --short=11 HEAD 2>/dev/null)
{ echo "[t373] host=$(hostname) commit=$HEAD dirty=$(git -C $W status --porcelain -uno | wc -l) test_md5=$(md5sum < $TF | cut -c1-32) timing=$TIMING $(date -u '+%F %T') UTC"
  env | grep -E '^(LTX|TT_|HF_|GEMMA|PYTHONPATH|RUN_|NUM_)' | sort; } | tee run.log
[ "$HEAD" = $WANT ] || { echo "[t373] wrong commit $HEAD != $WANT" | tee -a run.log; exit 3; }
[ -e $W/tmp/ltx_env_prewarm.yaml ] && { echo "[t373] $W/tmp/ltx_env_prewarm.yaml would change env" | tee -a run.log; exit 3; }
for f in $W/ttnn/ttnn/_ttnn.so $LTX_CHECKPOINT $GEMMA_PATH $TF $M/SHA256SUMS $M/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
  $M/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  $M/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  $M/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors $M/vae/ltx-2.5-audio-vae-bf16.safetensors; do
  [ -r $f ] || { echo "[t373] missing $f" | tee -a run.log; exit 3; }
done
case "$(readlink -f $LTX_CHECKPOINT) $(readlink -f $M) $(readlink -f $GEMMA_PATH)" in /mnt/*|*" /mnt/"*) echo "[t373] network fs path" | tee -a run.log; exit 3;; esac
T0=$(date +%s)
python -u -m pytest -c $W/pytest.ini --rootdir=$W -sv -p no:cacheprovider --timeout=540 \
  "$TF::test_pipeline_distilled[blackhole-bh_4x8sp1tp0_ring-True]" 2>&1 | tee -a run.log
rc=${PIPESTATUS[0]}
echo "[t373] process wall $(( $(date +%s) - T0 )) s" | tee -a run.log
echo "[t373] planar concat: $(python -c 'from models.tt_dit.utils.planar_concat import HAS_CPP_PLANAR_CONCAT as h; print("HAS_CPP_PLANAR_CONCAT", h)' 2>&1 | tail -1)" | tee -a run.log
echo "[t373] AICLK clamp warnings: $(grep -c "AICLK failed to settle" run.log)" | tee -a run.log
for mp4 in ltx_av_fast_*.mp4; do
  [ -e "$mp4" ] && ffmpeg -loglevel error -y -ss 3 -i "$mp4" -frames:v 1 "${mp4%.mp4}_t3s.png"
done
md5sum ltx_av_fast_*.mp4 2>/dev/null | tee -a run.log
echo "T373_EXIT=$rc" | tee -a run.log
exit $rc
