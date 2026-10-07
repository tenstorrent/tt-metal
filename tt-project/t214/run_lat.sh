#!/bin/bash
# Job A (blx01 broker): t48 5e4e0cd643a defaults, DEFAULT prompt, 1088x1920/145f, 4x8 ring, traced.
# One process: warmup, gen#0 (capture, seed 0), warm replays for seeds 0-4. LTX_DUMP_LATENT writes the
# normalized BCTHW latent handed to the VAE on every decode; map_latents.py names them by seed afterwards.
# Python from the t208 overlay tree (5e4e0cd643a, read only); C++ build + JIT cache from /var/tmp/fasth3/t48.
set -o pipefail
F=/var/tmp/fasth3; W=$F/t48; O=$F/t208/tree; D=$F/diffvae
OUT=$D/pipeline; rm -rf $OUT $D/latents_raw; mkdir -p $OUT
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface
mkdir -p $TMPDIR
source $W/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$O:$W:$W/ttnn:$W/tools HF_HUB_OFFLINE=1
export LTX_VERSION=2.5 LTX25_DIFFVAE=0
export LTX_CHECKPOINT=$F/models/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
export LTX25_VIDEO_VAE=$LTX_CHECKPOINT
export LTX25_ROOT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941
export FASTH3_DATA=$F GEMMA_PATH=$F/models/gemma-3-12b-it-qat-q4_0-unquantized
export NO_PROMPT=1 SEED=0 RUN_WARMUP=1 LTX_TRACED=1 RUN_VBENCH=0 RUN_CLIP=0
export LTX_FRESH_PROMPTS=0 LTX_E2E_SEEDS=0,1,2,3,4 LTX_E2E_EXTRA_REPLAYS=0 LTX_TIME_STAGES=1 LTX_CONV3D_BLOCKING_MESH=4,8
export LTX_DUMP_LATENT=$D/latents_raw
export TT_METAL_CACHE=$F/cache/tt-metal-cache TT_DIT_CACHE_DIR=$F/cache/dit-ltx25
export LTX_OUT_DIR=$OUT
cd $OUT
{
  echo "[t214] job A host=$(hostname) build=$(git -C $W rev-parse --short=11 HEAD) py=$(cat $O/OVERLAY_COMMIT) $(date -u '+%F %T')"
  env | grep -E '^(LTX|TT_DIT|TT_METAL_CACHE|NO_PROMPT|SEED|RUN_|GEMMA)' | sort
} | tee $OUT/run.log
[ "$(cat $O/OVERLAY_COMMIT)" = 5e4e0cd643a ] || { echo "T214A_EXIT=5 overlay not 5e4e0cd643a" | tee -a $OUT/run.log; exit 5; }
T0=$(date +%s)
python -u -m pytest -c $O/pytest.ini --rootdir=$O -sv -p no:cacheprovider --timeout=${PYTEST_S:-330} \
  "$O/models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled" \
  -k bh_4x8sp1tp0_ring 2>&1 | tee -a $OUT/run.log
rc=${PIPESTATUS[0]}
echo "[t214] process wall $(( $(date +%s) - T0 )) s, jit compiles $(grep -c 'BuildKernels | compiled' $OUT/run.log)" | tee -a $OUT/run.log
mv -f ltx_av_fast_*.mp4 $OUT/ 2>/dev/null
echo "T214A_EXIT=$rc" | tee -a $OUT/run.log
exit $rc
