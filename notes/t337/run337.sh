#!/bin/bash
# t337: the repo's standard LTX e2e test, unmodified, on ttp/t48-ltx25-integrated 63e54d98036 (own Release build), blx01
# 4x8, LTX_VERSION=2.5 LTX25_DIFFVAE=0, defaults otherwise (8+3, 1088x1920, 24 fps, 145 frames, seed 10, test prompt,
# bh_4x8sp1tp0_ring, traced). Usage: run337.sh <arm>: conv25 = the user's verified 2.5 conv VAE copy (#243) via
# LTX25_VIDEO_VAE; conv23 = LTX25_VIDEO_VAE set to the 2.3 monolith (same weights, md5 check). Own JIT t337/jit.
if [ -z "$INNER" ]; then
  INNER=1 setsid bash "$0" "$@" & PG=$!
  trap 'kill -TERM -- -$PG 2>/dev/null; sleep 5; kill -KILL -- -$PG 2>/dev/null' EXIT
  trap 'exit 143' TERM; trap 'exit 130' INT
  wait $PG; exit $?
fi
set -o pipefail
ARM=${1:?arm}
F=/var/tmp/fasth3; T=$F/t337; W=$T/b; WANT=63e54d98036; OUT=$T/out_$ARM; M=$F/models/ltx-2.5
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 70 ] || { echo "[t337] df / $use% > 70%"; exit 5; }
gb=$(timeout 120 du -sxBG $F | cut -f1 | tr -dc 0-9); [ "${gb:-999}" -le 150 ] || { echo "[t337] $F ${gb}G > 150G"; exit 5; }
rm -rf $OUT; mkdir -p $OUT $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp HF_HOME=$F/home/.cache/huggingface HF_HUB_OFFLINE=1
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools PYTHONDONTWRITEBYTECODE=1 TT_METAL_CACHE=$T/jit
export LTX_CHECKPOINT=$F/models/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
export GEMMA_PATH=$F/models/gemma-3-12b-it-qat-q4_0-unquantized
export LTX_VERSION=2.5 LTX25_ROOT=$M LTX25_DIFFVAE=0
unset LTX25_VIDEO_VAE LTX25_VAE_FALLBACK_23
case $ARM in conv25) export LTX25_VIDEO_VAE=$F/ltx25_vae/ltx-2.5-video-vae-conv-bf16.safetensors;; conv23) export LTX25_VIDEO_VAE=$LTX_CHECKPOINT;; *) echo "[t337] bad arm $ARM"; exit 3;; esac
export RUN_VBENCH=0 RUN_CLIP=0 LTX_OUT_DIR=$OUT
unset TT_DIT_CACHE_DIR LTX_FUSE_GATE_ON_DEVICE LTX_FUSE_NORM_ADALN LTX_QUANT LTX_QUANT_ACTIVATIONS LTX_QUALITY LTX_FAST \
  LTX_S1_SIGMAS LTX_S2_SIGMAS LTX_TRACED LTX_ITER_ENV NO_PROMPT SEED RUN_WARMUP PROMPT OUTPUT_PATH LTX25_TEXT_STACK \
  LTX_E2E_SEEDS LTX_E2E_EXTRA_REPLAYS LTX_FRESH_PROMPTS LTX_E2E_AB_ENV NUM_FRAMES HEIGHT WIDTH FPS DIFFVAE_SLAB_FRAMES
cd $OUT || exit 3
TF=$W/models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py
HEAD=$(git -C $W rev-parse --short=11 HEAD 2>/dev/null)
{ echo "[t337] host=$(hostname) commit=$HEAD dirty=$(git -C $W status --porcelain -uno | wc -l) test_md5=$(md5sum < $TF | cut -c1-32) arm=$ARM $(date -u '+%F %T') UTC"
  env | grep -E '^(LTX|TT_|HF_|GEMMA|PYTHONPATH|RUN_|NUM_)' | sort; } | tee run.log
[ "$HEAD" = $WANT ] || { echo "[t337] wrong commit $HEAD != $WANT" | tee -a run.log; exit 3; }
[ -e $W/tmp/ltx_env_prewarm.yaml ] && { echo "[t337] $W/tmp/ltx_env_prewarm.yaml would change env" | tee -a run.log; exit 3; }
for f in $W/ttnn/ttnn/_ttnn.so $LTX_CHECKPOINT $GEMMA_PATH $TF $M/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
  $M/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  $M/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  $M/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors $M/vae/ltx-2.5-audio-vae-bf16.safetensors; do
  [ -r $f ] || { echo "[t337] missing $f" | tee -a run.log; exit 3; }
done
case "$(readlink -f $LTX_CHECKPOINT) $(readlink -f $M)" in /mnt/*|*" /mnt/"*) echo "[t337] network fs path" | tee -a run.log; exit 3;; esac
T0=$(date +%s)
python -u -m pytest -c $W/pytest.ini --rootdir=$W -sv -p no:cacheprovider --timeout=540 \
  "$TF::test_pipeline_distilled[blackhole-bh_4x8sp1tp0_ring-True]" 2>&1 | tee -a run.log
rc=${PIPESTATUS[0]}
echo "[t337] process wall $(( $(date +%s) - T0 )) s" | tee -a run.log
echo "[t337] AICLK clamp warnings: $(grep -c "AICLK failed to settle" run.log)" | tee -a run.log
for mp4 in ltx_av_fast_*.mp4; do
  [ -e "$mp4" ] && ffmpeg -loglevel error -y -ss 3 -i "$mp4" -frames:v 1 "${mp4%.mp4}_t3s.png"
done
md5sum ltx_av_fast_*.mp4 2>/dev/null | tee -a run.log
echo "T337_EXIT=$rc" | tee -a run.log
exit $rc
