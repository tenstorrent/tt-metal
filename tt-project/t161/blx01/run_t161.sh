#!/bin/bash
# t161: one H3 fl2va broker job on blx01's t48 tree. Env: T161_MODE (capture|fill|time), T161_SECONDS, T161_TAG.
set -o pipefail
F=/var/tmp/fasth3
W=$F/t48
export T161_MODE=${T161_MODE:?} T161_SECONDS=${T161_SECONDS:?}
export T161_OUT=$F/t161/out_${T161_TAG:?} T161_FIRST=$F/t161/kf_first.png T161_LAST=$F/t161/kf_last.png
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface
export MINIMAX_H3_MODEL_PATH=/mnt/MLPerf/tt-shield/persistent-volume/volume_id_tt_transformers-MiniMax-H3-v0.22.0/weights/MiniMax-H3
# H3 gets its own kernel cache so its prewarm manifest does not mix with the LTX one.
export TT_METAL_CACHE=$F/cache/tt-metal-cache-h3 TT_DIT_CACHE_DIR=$F/cache/dit-h3 HF_HUB_OFFLINE=1
[ "$T161_MODE" = capture ] && export TT_METAL_KERNEL_PREWARM=1 TT_METAL_KERNEL_CAPTURE_ONLY=1
mkdir -p $TMPDIR $T161_OUT $TT_METAL_CACHE
cd $W || exit 3
source $W/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools
echo "[t161] host=$(hostname) commit=$(git -C $W rev-parse HEAD) mode=$T161_MODE seconds=$T161_SECONDS $(date -u '+%F %T') UTC" | tee $T161_OUT/run.log
T0=$(date +%s)
python -u -m pytest -sv -p no:cacheprovider -p conftest --rootdir=$W --timeout=570 \
  $F/t161/test_t161_h3_timing.py 2>&1 | tee -a $T161_OUT/run.log; rc=${PIPESTATUS[0]}
echo "[t161] process wall $(( $(date +%s) - T0 )) s" | tee -a $T161_OUT/run.log
grep -E 'T161_RESULT|T161 ' $T161_OUT/run.log | tail -5
echo "T161_EXIT=$rc" | tee -a $T161_OUT/run.log
exit $rc
