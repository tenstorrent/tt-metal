#!/bin/bash
# One eval-pack config as one blx01 broker job: 4x8 ring, 1088x1920/145f, traced, default warmup (job 621's;
# #166's warmup cuts dropped tray 3 twice on blx01). One pytest process: warmup, gen#0 (capture, default
# prompt), gen#1 (first warm replay, the headline), plus LTX_E2E_EXTRA_REPLAYS more warm replays.
# Python from the t171 overlay tree (t48 f6b806516cc), C++ build + kernels + JIT cache from /var/tmp/fasth3/t48 (bf7db12a149).
# pytest runs from $OUT (no kernel sources there), so relative kernel paths resolve to TT_METAL_HOME as in
# job 621. Everything stays under /var/tmp/fasth3: blx01 /home is full.
# Usage: bash run_cfg.sh <label> [VAR=val ...]   (PYTEST_S sizes the in-process timeout)
# t186: t171 tree (t48 f6b806516cc), one 5-seed job with an LTX_S1_SIGMAS override.
set -o pipefail
F=/var/tmp/fasth3; W=$F/t48; T=$F/t186; O=$F/t171/tree
label=$1; shift
OUT=$T/res/$label; rm -rf $OUT; mkdir -p $OUT
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
export LTX_E2E_EXTRA_REPLAYS=1 LTX_FRESH_PROMPTS=1 LTX_TIME_STAGES=1 LTX_CONV3D_BLOCKING_MESH=4,8
export TT_METAL_LOG_KERNEL_COMPILE=1
export TT_METAL_CACHE=$F/cache/tt-metal-cache TT_DIT_CACHE_DIR=$F/cache/dit-ltx25
for kv in "$@"; do export "$kv"; done
export LTX_OUT_DIR=$OUT
cd $OUT
{
  echo "[t186] label=$label flags=$* host=$(hostname) boot=$(uptime -s) build=$(git -C $W rev-parse --short=11 HEAD) py=$(cat $O/OVERLAY_COMMIT) $(date -u '+%F %T')"
  env | grep -E '^(LTX|TT_DIT|TT_METAL_CACHE|NO_PROMPT|SEED|RUN_|GEMMA)' | sort
} | tee $OUT/run.log
test -f $W/ttnn/ttnn/_ttnn.so || { echo "[t186] no build at $W" | tee -a $OUT/run.log; echo "T186_EXIT[$label]=4" | tee -a $OUT/run.log; exit 4; }
for f in $LTX_CHECKPOINT $LTX25_ROOT $GEMMA_PATH $TT_DIT_CACHE_DIR $O/conftest.py; do
  [ -e $f ] || { echo "missing $f" | tee -a $OUT/run.log; echo "T186_EXIT[$label]=3" | tee -a $OUT/run.log; exit 3; }
done
T0=$(date +%s)
python -u -m pytest -c $O/pytest.ini --rootdir=$O -sv -p no:cacheprovider --timeout=${PYTEST_S:-240} \
  "$O/models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled" \
  -k bh_4x8sp1tp0_ring 2>&1 | tee -a $OUT/run.log
rc=${PIPESTATUS[0]}
echo "[t186] process wall $(( $(date +%s) - T0 )) s, jit compiles $(grep -c 'BuildKernels | compiled' $OUT/run.log)" | tee -a $OUT/run.log
mv -f ltx_av_fast_*.mp4 $OUT/ 2>/dev/null
for mp4 in $OUT/*.mp4; do
  [ -e "$mp4" ] && ffmpeg -loglevel error -y -ss 3 -i "$mp4" -frames:v 1 -q:v 3 "${mp4%.mp4}_t3s.jpg"
done
grep -E 'E2E_WALL_S|Video export|done in [0-9.]+s$' $OUT/run.log | grep -E 'E2E|export|warmup' | tail -12
echo "T186_EXIT[$label]=$rc" | tee -a $OUT/run.log
exit $rc
