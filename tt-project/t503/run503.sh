#!/bin/bash
# t503 A/B pair on blx03 (one broker job, -t 600, unmeasured): the repo's standard LTX-2.5 e2e test, unmodified,
# arm old (stale MLPerf 28dac7ac encoder) then arm new (Aug-17 encoder, HF 1b92891ced, LTX25_TEXT_ENCODER_PATH).
# Code: origin/ltx-rt f6547442b30 + 94825eb729e + df0b2742038; C++ build copied from g15blx02 worktree t313 (same C++).
# LTX_VERSION=2.5 LTX25_DIFFVAE=0, 2.5 conv VAE, defaults otherwise (8+3, 1088x1920, 145 frames, seed 10).
# Resumable: an arm with out_<arm>/ARM_OK is skipped, and the DiT/encoder weight caches persist in $R/cache,
# so a rerun after a timeout continues warm. plug498 saves the prompt embeddings.
set -o pipefail
R=/var/tmp/fasth3/t503; W=/home/smarton/fasth3/t503/tt-metal; T0=$(date +%s); END=$((T0 + 575))
use=$(df --output=pcent / | tail -1 | tr -dc 0-9); [ "$use" -le 85 ] || { echo "[t503] / at ${use}%"; exit 5; }
gb=$(timeout 120 du -sxBG /var/tmp/fasth3 | cut -f1 | tr -dc 0-9); [ "${gb:-0}" -le 700 ] || { echo "[t503] /var/tmp/fasth3 ${gb}G"; exit 5; }
source /home/smarton/fasth3/tt-metal/python_env/bin/activate
export LD_LIBRARY_PATH=$W/build_Release/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools:$R/plug PYTHONDONTWRITEBYTECODE=1 TMPDIR=$R/tmp HF_HUB_OFFLINE=1
export TT_METAL_CACHE=$R/jit TT_DIT_CACHE_DIR=$R/cache
export LTX_VERSION=2.5 LTX25_DIFFVAE=0 RUN_VBENCH=0 RUN_CLIP=0 LTX25_ROOT=$R/ltx25
export LTX25_VIDEO_VAE=$R/ltx25/vae/ltx-2.5-video-vae-conv-bf16.safetensors
# the test's skip gate checks the 2.3 checkpoint even for LTX_VERSION=2.5 (the 2.5 branch does not read it)
export LTX_CHECKPOINT=$LTX25_VIDEO_VAE
unset LTX25_TEXT_ENCODER_PATH LTX_FUSE_GATE_ON_DEVICE LTX_QUANT LTX_QUANT_ACTIVATIONS LTX_QUALITY LTX_FAST LTX_S1_SIGMAS \
  LTX_S2_SIGMAS LTX_TRACED LTX_ITER_ENV NO_PROMPT SEED RUN_WARMUP PROMPT OUTPUT_PATH LTX_E2E_SEEDS LTX_E2E_EXTRA_REPLAYS \
  LTX_FRESH_PROMPTS LTX25_TEXT_STACK NUM_FRAMES HEIGHT WIDTH FPS DIFFVAE_SLAB_FRAMES
T=$W/models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py
for f in $W/ttnn/ttnn/_ttnn.so $W/build_Release/lib/_ttnncpp.so $T $R/plug/plug498.py $R/ltx25/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  $R/te0817/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors $LTX25_VIDEO_VAE; do [ -r $f ] || { echo "[t503] missing $f"; exit 3; }; done
# a quick (<150 s) failure in an earlier attempt is a bug, not a slow cold start: do not burn another job on it
[ -e $R/last_fail ] && { echo "[t503] earlier attempt failed fast: $(cat $R/last_fail)"; exit 9; }
rc=0
for ARM in old new; do
  OUT=$R/out_$ARM; [ -e $OUT/ARM_OK ] && { echo "[t503] arm $ARM already done"; continue; }
  left=$((END - $(date +%s))); [ $left -ge 60 ] || { echo "[t503] no time left for arm $ARM"; rc=7; break; }
  rm -rf $OUT; mkdir -p $OUT; cd $OUT || exit 3
  export T498_EMB_DIR=$OUT LTX_OUT_DIR=$OUT
  [ $ARM = new ] && export LTX25_TEXT_ENCODER_PATH=$R/te0817/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
  echo "[t503] arm=$ARM host=$(hostname) commit=$(cat $W/T503_COMMIT) $(date -u '+%F %T') UTC left=${left}s" | tee run.log
  env | grep -E '^(LTX|TT_|HF_|PYTHONPATH|LD_LIBRARY_PATH|RUN_|T498)' | sort >> run.log
  A0=$(date +%s)
  timeout --foreground -k 10 $((left - 5)) python -u -m pytest -c $W/pytest.ini --rootdir=$W -sv -p no:cacheprovider -p plug498 \
    --timeout=$((left - 15)) "$T::test_pipeline_distilled[blackhole-bh_4x8sp1tp0_ring-True]" 2>&1 | tee -a run.log; prc=${PIPESTATUS[0]}
  echo "[t503] arm $ARM process wall $(( $(date +%s) - A0 )) s rc=$prc" | tee -a run.log
  for mp4 in ltx_av_fast_*.mp4; do [ -e "$mp4" ] && ffmpeg -loglevel error -y -ss 3 -i "$mp4" -frames:v 1 "${mp4%.mp4}_t3s.png"; done
  md5sum ltx_av_fast_*.mp4 2>/dev/null | tee -a run.log
  if [ "$prc" = 0 ] && grep -qE "^=+ 1 passed" run.log; then touch ARM_OK; else rc=${prc/#0/6}; echo "[t503] arm $ARM did not pass" | tee -a run.log
    [ $(( $(date +%s) - A0 )) -lt 150 ] && echo "arm=$ARM rc=$prc wall=$(( $(date +%s) - A0 ))s" > $R/last_fail; break; fi
done
echo "T503_EXIT=$rc"; exit $rc
