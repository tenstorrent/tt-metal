#!/bin/bash
# t209: one H3 fl2va broker job on the fasth3-opt tree. Env: T209_TAG, T209_LOAD_ONLY, T209_STEPS, T209_WARM, T209_SEEDS.
set -o pipefail
F=/var/tmp/fasth3; T=$F/t209; W=$T/b
export BASE_OUT=$T/out_${T209_TAG:?} BASE_FIRST=$T/kf_first.png BASE_LAST=$T/kf_last.png
export BASE_SECONDS=10 BASE_HEIGHT=768 BASE_WIDTH=1344 BASE_UPSCALE=1920x1080
export BASE_VSA_SPARSITY=0.9 BASE_VSA_RING_GATHER=fused_kv
export BASE_STEPS=${T209_STEPS:-50} BASE_WARM_STEPS=${T209_WARM:-2} BASE_SEEDS=${T209_SEEDS:-0,1} BASE_LOAD_ONLY=${T209_LOAD_ONLY:-0}
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp TORCH_HOME=$F/home/.cache/torch HF_HOME=$F/home/.cache/huggingface
export MINIMAX_H3_MODEL_PATH=/mnt/MLPerf/tt-shield/persistent-volume/volume_id_tt_transformers-MiniMax-H3-v0.22.0/weights/MiniMax-H3
export TT_METAL_CACHE=$F/cache/tt-metal-cache-h3opt TT_DIT_CACHE_DIR=$F/cache/dit-h3opt HF_HUB_OFFLINE=1
[ "${T209_MGD:-0}" = 1 ] && export TT_MESH_GRAPH_DESC_PATH=$W/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
mkdir -p $TMPDIR $BASE_OUT $TT_METAL_CACHE $TT_DIT_CACHE_DIR
cd $W || exit 3
source $F/t48/python_env/bin/activate
export TT_METAL_HOME=$W PYTHONPATH=$W:$W/ttnn:$W/tools
echo "[t209] host=$(hostname) commit=$(git -C $W rev-parse HEAD) tag=$T209_TAG steps=$BASE_STEPS warm=$BASE_WARM_STEPS seeds=$BASE_SEEDS load_only=$BASE_LOAD_ONLY $(date -u '+%F %T') UTC" | tee $BASE_OUT/run.log
env | grep -E '^(BASE_|TT_|VSA_|MINIMAX)' | sort >> $BASE_OUT/run.log
T0=$(date +%s)
python -u -m pytest -sv -p no:cacheprovider --timeout=${T209_PYTEST_TIMEOUT:-570} \
  models/tt_dit/tests/models/minimax_h3/test_fasth3_baseline_minimax_h3.py 2>&1 | tee -a $BASE_OUT/run.log; rc=${PIPESTATUS[0]}
echo "[t209] process wall $(( $(date +%s) - T0 )) s" | tee -a $BASE_OUT/run.log
echo "T209_EXIT=$rc" | tee -a $BASE_OUT/run.log
exit $rc
