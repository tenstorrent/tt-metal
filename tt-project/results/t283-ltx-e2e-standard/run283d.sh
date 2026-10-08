#!/bin/bash
# t283 attempt 4: as run283b.sh, plus P=<prec>nf sets LTX_FUSE_GATE_ON_DEVICE=0 (t48's default on-device gate fold
# breaks a cold-cache bf16 save and bf8 load, jobs 060/062). Pure ltx-rt HEAD fails unmodified (job 066).
# t283: the standard LTX e2e test, unmodified, on t48 (ttp/t48-ltx25-integrated). The ltx-rt HEAD copy fails
# unmodified (job 053: LTXPipeline.__init__ has no image_conditioning kwarg). Python = t208 overlay tree
# (t48 5e4e0cd643a; test file md5-identical to t48 HEAD 9e20d905481, later commits touch only DiffVAE),
# C++ build + kernels + JIT cache = /var/tmp/fasth3/t48. Test defaults (LTX-2.3, 1088x1920, 145f, seed 10,
# traced, NO_PROMPT gen#0 capture + gen#1 replay); only paths are set, and VBench/CLIP are off (RUN_VBENCH=0
# RUN_CLIP=0, the test's perf-only switch). Usage: run283b.sh bf16|bf8 (bf8 = LTX_QUANT=all_bf8_lofi).
set -o pipefail
P=${1:?bf16|bf8|bf16nf|bf8nf}; F=/var/tmp/fasth3; W=$F/t48; O=$F/t208/tree; OUT=$F/t283/out48d_$P
rm -rf $OUT; mkdir -p $OUT $F/t283/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/t283/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=/home/sulphur/hf
source $W/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$O:$W:$W/ttnn:$W/tools HF_HUB_OFFLINE=1
export LTX_CHECKPOINT=$F/models/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
export GEMMA_PATH=$F/models/gemma-3-12b-it-qat-q4_0-unquantized
export TT_DIT_CACHE_DIR=$F/t220/cache/dit-ltx23 TT_METAL_CACHE=$F/cache/tt-metal-cache
export RUN_VBENCH=0 RUN_CLIP=0 LTX_OUT_DIR=$OUT
unset LTX_FUSE_GATE_ON_DEVICE LTX_QUANT LTX_QUALITY LTX_FAST LTX_S1_SIGMAS LTX_S2_SIGMAS LTX_TRACED LTX_VERSION LTX_ITER_ENV NO_PROMPT SEED RUN_WARMUP
case $P in bf8*) export LTX_QUANT=all_bf8_lofi;; esac
case $P in *nf) export LTX_FUSE_GATE_ON_DEVICE=0;; esac
cd $OUT || exit 3
T=$O/models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py
{ echo "[t283] host=$(hostname) build=$(git -C $W rev-parse --short=11 HEAD) py=$(cat $O/OVERLAY_COMMIT) test_md5=$(md5sum < $T | cut -c1-32) prec=$P $(date -u '+%F %T') UTC"
  env | grep -E '^(LTX_|TT_|HF_|GEMMA|NUM_FRAMES|HEIGHT|WIDTH|FPS|SEED|RUN_|NO_PROMPT)' | sort; } | tee run.log
for f in $W/ttnn/ttnn/_ttnn.so $LTX_CHECKPOINT $GEMMA_PATH $TT_DIT_CACHE_DIR; do [ -e $f ] || { echo "missing $f" | tee -a run.log; exit 3; }; done
T0=$(date +%s)
python -u -m pytest -c $O/pytest.ini --rootdir=$O -sv -p no:cacheprovider --timeout=570 "$T::test_pipeline_distilled" -k bh_4x8sp1tp0_ring 2>&1 | tee -a run.log
rc=${PIPESTATUS[0]}
echo "[t283] process wall $(( $(date +%s) - T0 )) s" | tee -a run.log
echo "T283_EXIT=$rc" | tee -a run.log
exit $rc
