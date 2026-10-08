#!/bin/bash
# t295: t48 @ f6547442b30 (S2 default back to 3 steps) at its own defaults, no sigma env vars, 5 seeds in one
# pytest process: warmup, gen#0 cold seed 0, gen#1..5 warm seeds 0..4. Same env as ref_t48_f6b8 (job 710) and
# t188 job 755 (t171 run_cfg.sh), except LTX25_ROOT must be a LOCAL copy (no /mnt/MLPerf reads in device jobs).
# Python from the overlay tree $T/tree (setup295.sh), build + kernels + JIT cache from $W.
# Usage (broker -w $W -t 240; job 755 took 161.6 s): bash run295.sh <label> <local LTX25_ROOT>
set -o pipefail
F=/var/tmp/fasth3; W=$F/t48; T=$F/t295; O=$T/tree
label=$1; L25=$2
OUT=$T/res/$label; rm -rf $OUT; mkdir -p $OUT
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface
mkdir -p $TMPDIR
source $W/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$O:$W:$W/ttnn:$W/tools HF_HUB_OFFLINE=1
export LTX_VERSION=2.5 LTX25_DIFFVAE=0
export LTX_CHECKPOINT=$F/models/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
export LTX25_VIDEO_VAE=$LTX_CHECKPOINT LTX25_ROOT=$L25
export FASTH3_DATA=$F GEMMA_PATH=$F/models/gemma-3-12b-it-qat-q4_0-unquantized
export NO_PROMPT=1 SEED=0 RUN_WARMUP=1 LTX_TRACED=1 RUN_VBENCH=0 RUN_CLIP=0
export LTX_E2E_EXTRA_REPLAYS=0 LTX_FRESH_PROMPTS=0 LTX_E2E_SEEDS=0,1,2,3,4 LTX_TIME_STAGES=1 LTX_CONV3D_BLOCKING_MESH=4,8
export TT_METAL_LOG_KERNEL_COMPILE=1
export TT_METAL_CACHE=$F/cache/tt-metal-cache TT_DIT_CACHE_DIR=$F/cache/dit-ltx25
unset LTX_S1_SIGMAS LTX_S2_SIGMAS LTX_QUALITY LTX_FAST LTX_QUANT
export LTX_OUT_DIR=$OUT
cd $OUT
{
  echo "[t295] label=$label host=$(hostname) boot=$(uptime -s) build=$(git -C $W rev-parse --short=11 HEAD) py=$(cat $O/OVERLAY_COMMIT) $(date -u '+%F %T')"
  env | grep -E '^(LTX|TT_DIT|TT_METAL_CACHE|NO_PROMPT|SEED|RUN_|GEMMA)' | sort
} | tee $OUT/run.log
case "$L25" in /mnt/*) echo "LTX25_ROOT on network fs" | tee -a $OUT/run.log; echo "T295_EXIT=5" | tee -a $OUT/run.log; exit 5;; esac
for f in $W/ttnn/ttnn/_ttnn.so $LTX_CHECKPOINT $LTX25_ROOT $GEMMA_PATH $TT_DIT_CACHE_DIR $O/conftest.py; do
  [ -e $f ] || { echo "missing $f" | tee -a $OUT/run.log; echo "T295_EXIT=3" | tee -a $OUT/run.log; exit 3; }
done
T0=$(date +%s)
python -u -m pytest -c $O/pytest.ini --rootdir=$O -sv -p no:cacheprovider --timeout=230 \
  "$O/models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled" \
  -k bh_4x8sp1tp0_ring 2>&1 | tee -a $OUT/run.log
rc=${PIPESTATUS[0]}
echo "[t295] process wall $(( $(date +%s) - T0 )) s, jit compiles $(grep -c 'BuildKernels | compiled' $OUT/run.log)" | tee -a $OUT/run.log
mv -f ltx_av_fast_*.mp4 ltx_av_fast_*.json $OUT/ 2>/dev/null
grep -E 'E2E_WALL_S|sigma|steps' $OUT/run.log | tail -12
md5sum $OUT/*.mp4 | tee -a $OUT/run.log
echo "T295_EXIT=$rc" | tee -a $OUT/run.log
exit $rc
