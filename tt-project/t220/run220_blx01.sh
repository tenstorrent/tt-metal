#!/bin/bash
# t251: t220 on blx01. LTX-2.3 (ltx-rt b9f8587ce6c + test patch, tree d791f4e949) 8-bit = production
# LTX_QUALITY=medium, BH 4x8 ring, 1088x1920 145f. Everything under /var/tmp/fasth3 (blx01 /home is full).
# Env: T220_TAG (out dir), T220_SEEDS (extra warm seeds after seed 0, e.g. 1,2,3,4), T220_PYTEST_S.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t220; W=$T/src; OUT=$T/out_${T220_TAG:?}
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$T/tmp
export HF_HOME=/home/sulphur/hf HF_HUB_OFFLINE=1 GEMMA_PATH=google/gemma-3-12b-it-qat-q4_0-unquantized
export LTX_CHECKPOINT=/home/sulphur/hf/hub/models--Lightricks--LTX-2.3/snapshots/76730e634e70a28f4e8d51f5e29c08e40e2d8e74/ltx-2.3-22b-distilled-1.1.safetensors
export TT_DIT_CACHE_DIR=$T/cache/dit-ltx23 TT_METAL_CACHE=$T/cache/tt-metal-cache-ltx23
# Production worker env for the medium tier on the galaxy (ltx_server build_worker_env @ aadb386).
export LTX_QUALITY=medium LTX_TRACED=1 TT_DIT_HOST_WEIGHT_CACHE=1 TT_METAL_KERNEL_PREWARM=1
export LTX_ATTN_FABRIC_AGMM=0 LTX_BWE_TRACE=0 LTX_VOC_TRACE=0 TT_METAL_OPERATION_TIMEOUT_SECONDS=180 PYTHONFAULTHANDLER=1
export NUM_FRAMES=145 HEIGHT=1088 WIDTH=1920 FPS=24 SEED=0 LTX_E2E_SEEDS=${T220_SEEDS:-} RUN_CLIP=0 RUN_VBENCH=0 NO_PROMPT=1
mkdir -p $HOME $TMPDIR $OUT $TT_METAL_CACHE
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$W TT_METAL_RUNTIME_ROOT=$W PYTHONPATH=$W:$W/ttnn:$W/tools
cd $OUT || exit 3
echo "[t220] host=$(hostname) commit=$(cat $W/COMMIT) tag=$T220_TAG seeds=0,${T220_SEEDS} $(date -u '+%F %T') UTC" | tee run.log
env | grep -E '^(LTX_|TT_|HF_|GEMMA|NUM_FRAMES|HEIGHT|WIDTH|FPS|SEED|RUN_)' | sort >> run.log
T0=$(date +%s)
python -u -m pytest -sv -p no:cacheprovider --timeout=${T220_PYTEST_S:-570} \
  "$W/models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled" -k bh_4x8sp1tp0_ring 2>&1 | tee -a run.log; rc=${PIPESTATUS[0]}
echo "[t220] process wall $(( $(date +%s) - T0 )) s" | tee -a run.log
echo "T220_EXIT=$rc" | tee -a run.log
exit $rc
