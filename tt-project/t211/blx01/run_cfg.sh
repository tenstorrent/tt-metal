#!/bin/bash
# t211: one LTX-2.5 4x8 1080p/145f gen of seed 0, DEFAULT prompt, at t48 HEAD defaults (5e4e0cd643a python from the
# t208 overlay tree, read only; C++ build, kernels and JIT cache from /var/tmp/fasth3/t48). The two arms differ only in
# LTX25_DIFFVAE (0 = 2.3 conv VAE swap, 1 = 2.5 DiffVAE). LTX_DUMP_LATENTS saves each generate()'s denoised S2
# latents so the driver can prove both arms decoded the same latents. Copy of t208's run_cfg.sh otherwise.
# Usage: bash run_cfg.sh <label> [VAR=val ...]   (PYTEST_S sizes the in-process timeout)
set -o pipefail
F=/var/tmp/fasth3; W=$F/t48; T=$F/t211; O=$F/t208/tree
label=$1; shift
OUT=$T/res/$label; rm -rf $OUT; mkdir -p $OUT
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface
mkdir -p $TMPDIR
source $W/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$O:$W:$W/ttnn:$W/tools HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export LTX_VERSION=2.5 LTX25_DIFFVAE=0
export LTX_CHECKPOINT=$F/models/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
export LTX25_VIDEO_VAE=$LTX_CHECKPOINT
export LTX25_ROOT=/mnt/MLPerf/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/28dac7acdc1f78a70e98687db261a949754f8941
export FASTH3_DATA=$F GEMMA_PATH=$F/models/gemma-3-12b-it-qat-q4_0-unquantized
export NO_PROMPT=1 SEED=0 RUN_WARMUP=1 LTX_TRACED=1 RUN_VBENCH=0 RUN_CLIP=0
export LTX_E2E_SEEDS=0 LTX_E2E_EXTRA_REPLAYS=0 LTX_FRESH_PROMPTS=0 LTX_TIME_STAGES=1 LTX_CONV3D_BLOCKING_MESH=4,8
export TT_METAL_LOG_KERNEL_COMPILE=1
export TT_METAL_CACHE=$F/cache/tt-metal-cache TT_DIT_CACHE_DIR=$F/cache/dit-ltx25
for kv in "$@"; do export "$kv"; done
export LTX_OUT_DIR=$OUT LTX_DUMP_LATENTS=$OUT/s2lat
cd $OUT
{
  echo "[t211] label=$label flags=$* host=$(hostname) boot=$(uptime -s) build=$(git -C $W rev-parse --short=11 HEAD) py=$(cat $O/OVERLAY_COMMIT) $(date -u '+%F %T')"
  env | grep -E '^(LTX|TT_DIT|TT_METAL_CACHE|NO_PROMPT|SEED|RUN_|GEMMA|DIFFVAE)' | sort
} | tee $OUT/run.log
test -f $W/ttnn/ttnn/_ttnn.so || { echo "[t211] no build at $W" | tee -a $OUT/run.log; echo "T211_EXIT[$label]=4" | tee -a $OUT/run.log; exit 4; }
for f in $LTX_CHECKPOINT $LTX25_ROOT $LTX25_ROOT/vae/ltx-2.5-video-vae-bf16.safetensors $GEMMA_PATH $TT_DIT_CACHE_DIR $O/conftest.py; do
  [ -e $f ] || { echo "missing $f" | tee -a $OUT/run.log; echo "T211_EXIT[$label]=3" | tee -a $OUT/run.log; exit 3; }
done
T0=$(date +%s)
python -u -m pytest -c $O/pytest.ini --rootdir=$O -sv -p no:cacheprovider --timeout=${PYTEST_S:-200} \
  "$O/models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled" \
  -k bh_4x8sp1tp0_ring 2>&1 | tee -a $OUT/run.log
rc=${PIPESTATUS[0]}
echo "[t211] process wall $(( $(date +%s) - T0 )) s, jit compiles $(grep -c 'BuildKernels | compiled' $OUT/run.log)" | tee -a $OUT/run.log
mv -f ltx_av_fast_*.mp4 $OUT/ 2>/dev/null
grep -E 'E2E_WALL_S|VAE decode|LTX_DUMP_LATENTS' $OUT/run.log | tail -12
echo "T211_EXIT[$label]=$rc" | tee -a $OUT/run.log
exit $rc
