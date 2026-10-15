#!/bin/bash
# t511 blx03 job (one broker job): the repo's standard LTX-2.5 e2e test on origin/ltx-rt f6547442b30 + 923fffb239d
# (cherry-pick of a22dc027ee2, LTX25_S2_ANCESTRAL opt-in). Tree ~/fasth3/t511/tt-metal = hardlinked copy of the t503 tree
# (C++ build from g15blx02 worktree t313, same C++ as f6547442b30) with models/ and runtime/hw/lib from g15blx02.
# LTX_VERSION=2.5 LTX25_DIFFVAE=0, 2.5 conv VAE, weights from /var/tmp/fasth3/t503/ltx25 (local, read-only).
# Usage: run511.sh e2e  -> test defaults (8+3, 1088x1920, 145 frames, seed 10, traced, gen #0 capture + gen #1 replay)
#        run511.sh ab   -> SEED=0, gen #0 capture (off), gen #1 replay off, gen #2 replay LTX25_S2_ANCESTRAL=1 (LTX_E2E_AB_ENV)
set -o pipefail
MODE=${1:?mode}; R=/var/tmp/fasth3/t511; W=/home/smarton/fasth3/t511/tt-metal; L=/var/tmp/fasth3/t503/ltx25
T0=$(date +%s); END=$((T0 + ${T511_BUDGET:-575}))
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 85 ] || { echo "[t511] / at ${use}%"; exit 5; }
gb=$(timeout 120 du -sxBG /var/tmp/fasth3 | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le 700 ] || { echo "[t511] /var/tmp/fasth3 ${gb}G"; exit 5; }
source /home/smarton/fasth3/tt-metal/python_env/bin/activate
export LD_LIBRARY_PATH=$W/build_Release/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools PYTHONDONTWRITEBYTECODE=1 TMPDIR=$R/tmp HF_HUB_OFFLINE=1
export TT_METAL_CACHE=$R/jit
export LTX_VERSION=2.5 LTX25_DIFFVAE=0 RUN_VBENCH=0 RUN_CLIP=0 LTX25_ROOT=$L
# ltx-rt's VAE loaders read only vae.-prefixed keys; the 2.5 split file has bare decoder./encoder. keys (fixed on
# t48 by 63e54d98036, not on ltx-rt). Same 170 tensors and metadata, keys prefixed with vae.
export LTX25_VIDEO_VAE=$R/vae/ltx-2.5-video-vae-conv-bf16-vaeprefix.safetensors
# the test's skip gate checks the 2.3 checkpoint even for LTX_VERSION=2.5 (the 2.5 branch does not read it)
export LTX_CHECKPOINT=$LTX25_VIDEO_VAE
unset TT_DIT_CACHE_DIR LTX25_TEXT_ENCODER_PATH LTX_FUSE_GATE_ON_DEVICE LTX_QUANT LTX_QUANT_ACTIVATIONS LTX_QUALITY LTX_FAST \
  LTX_S1_SIGMAS LTX_S2_SIGMAS LTX_TRACED LTX_ITER_ENV NO_PROMPT SEED RUN_WARMUP PROMPT OUTPUT_PATH LTX_E2E_SEEDS LTX_E2E_AB_ENV \
  LTX_E2E_EXTRA_REPLAYS LTX_FRESH_PROMPTS LTX25_TEXT_STACK NUM_FRAMES HEIGHT WIDTH FPS DIFFVAE_SLAB_FRAMES LTX25_S2_ANCESTRAL
case $MODE in
  e2e) ;;
  ab) export SEED=0 LTX_E2E_SEEDS=0,0+ LTX_E2E_AB_ENV=LTX25_S2_ANCESTRAL=1;;
  *) echo "[t511] bad mode $MODE"; exit 3;;
esac
T=$W/models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py
for f in $W/ttnn/ttnn/_ttnn.so $W/build_Release/lib/_ttnncpp.so $W/runtime/hw/lib/blackhole/substitutes.o $T \
  $L/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors $L/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  $L/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors $LTX25_VIDEO_VAE; do
  [ -r $f ] || { echo "[t511] missing $f"; exit 3; }; done
OUT=$R/out_$MODE; rm -rf $OUT; mkdir -p $OUT $R/tmp; cd $OUT || exit 3
export LTX_OUT_DIR=$OUT
left=$((END - $(date +%s)))
echo "[t511] mode=$MODE host=$(hostname) commit=$(cat $W/T511_COMMIT) $(date -u '+%F %T') UTC left=${left}s" | tee run.log
env | grep -E '^(LTX|TT_|HF_|PYTHONPATH|LD_LIBRARY_PATH|RUN_|SEED)' | sort >> run.log
A0=$(date +%s)
timeout --foreground -k 10 $((left - 5)) python -u -m pytest -c $W/pytest.ini --rootdir=$W -sv -p no:cacheprovider \
  --timeout=$((left - 15)) "$T::test_pipeline_distilled[blackhole-bh_4x8sp1tp0_ring-True]" 2>&1 | tee -a run.log; prc=${PIPESTATUS[0]}
echo "[t511] mode $MODE process wall $(( $(date +%s) - A0 )) s rc=$prc" | tee -a run.log
for mp4 in ltx_av_fast_*.mp4; do [ -e "$mp4" ] && ffmpeg -loglevel error -y -ss 3 -i "$mp4" -frames:v 1 "${mp4%.mp4}_t3s.png"; done
md5sum ltx_av_fast_*.mp4 2>/dev/null | tee -a run.log
if [ "$prc" = 0 ] && grep -qE "^=+ 1 passed" run.log; then echo "T511_EXIT=0"; exit 0; fi
echo "[t511] mode $MODE did not pass" | tee -a run.log; echo "T511_EXIT=${prc/#0/6}"; exit ${prc/#0/6}
