#!/bin/bash
# blx03 broker job for #111: LTX-2.5 e2e on the 2x4 submesh of the full mesh (test opens (4,8), then
# create_submesh(2,4)), 544x960/145f, traced, fresh prompt per gen. gen0 captures; gens 1..4 replay with
# LTX_PROMPT_HOST_COPY=1 LTX_LATENT_STATS=0 on gens 2 and 4 (gen 2 also runs the prompt bit-identity check).
# Python from the staged overlay $S; C++ build and kernels from $B (same C++ as this branch).
BASE=/home/smarton/fasth3/tt-metal; B=${B:-/home/smarton/fasth3/t48}; V=/var/tmp/fasth3/t111; S=$V/src
LOG=$V/run111.log
source $BASE/python_env/bin/activate
export TT_METAL_HOME=$B PYTHONPATH=$S:$B/ttnn:$B/tools HF_HUB_OFFLINE=1
export LTX_VERSION=2.5 LTX25_DIFFVAE=0
export LTX_CHECKPOINT=/home/smarton/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
export LTX25_VIDEO_VAE=$LTX_CHECKPOINT
export LTX25_ROOT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941
export GEMMA_PATH=/var/tmp/fasth3/models/gemma-3-12b-it-qat-q4_0-unquantized
export NO_PROMPT=1 SEED=0 RUN_WARMUP=0 LTX_TRACED=1 RUN_VBENCH=0 RUN_CLIP=0
export HEIGHT=544 WIDTH=960 NUM_FRAMES=145 LTX_TIME_STAGES=1 LTX_FRESH_PROMPTS=1 LTX_E2E_EXTRA_REPLAYS=3
export LTX_E2E_AB_ENV="LTX_PROMPT_HOST_COPY=1 LTX_LATENT_STATS=0" LTX_E2E_AB_ENV_ONCE="LTX_PROMPT_STAGING_CHECK=1"
export TT_METAL_CACHE=/var/tmp/fasth3/cache/tt-metal-cache TT_DIT_CACHE_DIR=/var/tmp/fasth3/cache/dit-ltx25
export LTX_OUT_DIR=$V/out
mkdir -p $LTX_OUT_DIR
cd $S
echo "[t111] build=$(git -C $B rev-parse --short HEAD) src=$(cat $S/REV)" | tee $LOG
test -f $B/ttnn/ttnn/_ttnn.so || { echo "[t111] no build at $B" | tee -a $LOG; exit 4; }
timeout 1700 python -u -m pytest -c $S/pytest.ini --rootdir=$S -sv --timeout=1650 \
  "models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled" \
  -k bh_4x8sub2x4sp1tp0 2>&1 | tee -a $LOG
rc=${PIPESTATUS[0]}
rm -f $S/ltx_av_fast_*.mp4
echo "T111_EXIT=$rc" | tee -a $LOG
exit $rc
