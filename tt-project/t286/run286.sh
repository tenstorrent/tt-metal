#!/bin/bash
# t286: one broker job running the unmodified Turbo e2e test (test_pipeline_turbo_minimax_h3.py) on
# ttp/fasth3-hyperflow (e24a2b93d79), fl2va, 4x8. Args: <tag> <duration 5|10|15>.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t286; W=$F/t284/b
TAG=${1:?tag}; DUR=${2:?duration}
OUT=$T/out_$TAG; mkdir -p $OUT $F/tmp
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface HF_HUB_OFFLINE=1
export MINIMAX_H3_MODEL_PATH=/mnt/MLPerf/tt-shield/persistent-volume/volume_id_tt_transformers-MiniMax-H3-v0.22.0/weights/MiniMax-H3
export MINIMAX_H3_TURBO_LORA_PATH=$F/models/lightx2v-h3-turbo/minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors
export MINIMAX_H3_TURBO_KEYFRAME=$F/t209/kf_first.png MINIMAX_H3_TURBO_TASK=fl2va MINIMAX_H3_TURBO_POINT=768p MINIMAX_H3_TURBO_NFE=4
export TT_METAL_CACHE=$F/cache/tt-metal-cache-h3hf TT_DIT_CACHE_DIR=$F/cache/dit-h3hf
mkdir -p $TT_METAL_CACHE $TT_DIT_CACHE_DIR
cd $W || exit 3
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools
if [ "$DUR" = 5 ]; then K="5s and not 15s and not 4x32 and not WH"; else K="${DUR}s and not 4x32 and not WH"; fi
CMD="python -u -m pytest -sv -p no:cacheprovider models/tt_dit/tests/models/minimax_h3/test_pipeline_turbo_minimax_h3.py -k '$K'"
echo "[t286] host=$(hostname) commit=$(git -C $W rev-parse HEAD) tag=$TAG dur=$DUR job=${TTP_RUNNER_JOB:-} $(date -u '+%F %T') UTC" | tee $OUT/run.log
echo "[t286] cmd: $CMD" | tee -a $OUT/run.log
env | grep -E '^(MINIMAX|TT_)' | sort >> $OUT/run.log
T0=$(date +%s)
eval "$CMD" 2>&1 | tee -a $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[t286] process wall $(( $(date +%s) - T0 )) s" | tee -a $OUT/run.log
cp -p $HOME/h3_turbo_artifacts/fl2va_turbo_1344x768_${DUR}s_4fwd.mp4 $OUT/ 2>/dev/null
echo "T286_EXIT=$rc" | tee -a $OUT/run.log
exit $rc
