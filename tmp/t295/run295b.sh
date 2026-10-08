#!/bin/bash
# t295 one arm, pure LTX-2.3 (t283 job 077 env: t220 bf16 DiT cache, t48 build bf7db12a14, gate fold default),
# standard e2e test unmodified, 8+3 via the arm: A = treeA defaults (no sigma env), B = treeB (pre-revert tip)
# + LTX_S2_SIGMAS=0.909375,0.725,0.421875,0.0. gen#0 seed 0 (capture), gen#1..5 replays seeds 0..4.
# Usage: run295b.sh A|B
set -o pipefail
A=${1:?A|B}; F=/var/tmp/fasth3; W=$F/t48; O=$F/t295/tree$A; OUT=$F/t295/out$A
rm -rf $OUT; mkdir -p $OUT $F/t295/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/t295/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=/home/sulphur/hf
source $W/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$O:$W:$W/ttnn:$W/tools HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export LTX_CHECKPOINT=$F/models/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
export GEMMA_PATH=$F/models/gemma-3-12b-it-qat-q4_0-unquantized
export TT_DIT_CACHE_DIR=$F/t220/cache/dit-ltx23 TT_METAL_CACHE=$F/cache/tt-metal-cache
export RUN_VBENCH=0 RUN_CLIP=0 LTX_OUT_DIR=$OUT SEED=0 LTX_E2E_SEEDS=0,1,2,3,4
unset LTX_FUSE_GATE_ON_DEVICE LTX_QUANT LTX_QUALITY LTX_FAST LTX_S1_SIGMAS LTX_S2_SIGMAS LTX_TRACED LTX_VERSION LTX_ITER_ENV NO_PROMPT RUN_WARMUP
[ $A = B ] && export LTX_S2_SIGMAS=0.909375,0.725,0.421875,0.0
cd $OUT || exit 3
T=$O/models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py
{ echo "[t295] arm=$A host=$(hostname) build=$(git -C $W rev-parse --short=11 HEAD) py=$(cat $O/OVERLAY_COMMIT) $(date -u '+%F %T') UTC"
  grep -n '^_DEFAULT_S2_SIGMAS' $O/models/tt_dit/pipelines/ltx/pipeline_ltx_distilled.py
  env | grep -E '^(LTX_|TT_|GEMMA|SEED|RUN_)' | sort; } | tee run.log
for f in $W/ttnn/ttnn/_ttnn.so $LTX_CHECKPOINT $GEMMA_PATH $TT_DIT_CACHE_DIR $T; do [ -e $f ] || { echo "missing $f" | tee -a run.log; exit 3; }; done
T0=$(date +%s)
python -u -m pytest -c $O/pytest.ini --rootdir=$O -sv -p no:cacheprovider --timeout=260 "$T::test_pipeline_distilled" -k bh_4x8sp1tp0_ring 2>&1 | tee -a run.log
rc=${PIPESTATUS[0]}
echo "[t295] arm=$A process wall $(( $(date +%s) - T0 )) s" | tee -a run.log
(cd $OUT && md5sum ltx_av_fast_*.mp4) | tee -a run.log
echo "T295_EXIT=$rc" | tee -a run.log
exit $rc
