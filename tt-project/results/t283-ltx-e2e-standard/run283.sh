#!/bin/bash
# t283: repo's standard LTX e2e test, unmodified (ltx-rt b9f8587ce6c test_pipeline_ltx_distilled.py), BH 4x8 ring.
# Arg 1: bf16 | bf8. Only difference: LTX_QUANT unset (test default = bf16/HiFi2) vs LTX_QUANT=all_bf8_lofi.
# Everything else is the test's default (8+3 steps, 1088x1920 145f, seed 10, traced), except RUN_VBENCH=0 RUN_CLIP=0.
set -o pipefail
P=${1:?bf16|bf8}; F=/var/tmp/fasth3; W=$F/t220/src; C=$F/t220/cache; OUT=$F/t283/out_$P
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/t283/tmp
export HF_HOME=/home/sulphur/hf HF_HUB_OFFLINE=1 GEMMA_PATH=google/gemma-3-12b-it-qat-q4_0-unquantized
export LTX_CHECKPOINT=/home/sulphur/hf/hub/models--Lightricks--LTX-2.3/snapshots/76730e634e70a28f4e8d51f5e29c08e40e2d8e74/ltx-2.3-22b-distilled-1.1.safetensors
export TT_DIT_CACHE_DIR=$C/dit-ltx23 TT_METAL_CACHE=$C/tt-metal-cache-ltx23
export RUN_VBENCH=0 RUN_CLIP=0
unset LTX_QUANT LTX_QUALITY LTX_FAST LTX_S1_SIGMAS LTX_S2_SIGMAS LTX_TRACED TT_DIT_HOST_WEIGHT_CACHE LTX_ITER_ENV
[ "$P" = bf8 ] && export LTX_QUANT=all_bf8_lofi
mkdir -p $HOME $TMPDIR $OUT $TT_METAL_CACHE
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$W TT_METAL_RUNTIME_ROOT=$W PYTHONPATH=$W:$W/ttnn:$W/tools
cd $OUT || exit 3
T=models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py
echo "[t283] host=$(hostname) tree=$(cat $W/COMMIT) test_md5=$(md5sum < $W/$T | cut -c1-32) prec=$P $(date -u '+%F %T') UTC" | tee run.log
env | grep -E '^(LTX_|TT_|HF_|GEMMA|NUM_FRAMES|HEIGHT|WIDTH|FPS|SEED|RUN_)' | sort >> run.log
T0=$(date +%s)
python -u -m pytest -sv -p no:cacheprovider --timeout=570 "$W/$T::test_pipeline_distilled" -k bh_4x8sp1tp0_ring 2>&1 | tee -a run.log; rc=${PIPESTATUS[0]}
echo "[t283] process wall $(( $(date +%s) - T0 )) s" | tee -a run.log
echo "T283_EXIT=$rc" | tee -a run.log
exit $rc
