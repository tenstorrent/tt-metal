#!/bin/bash
# E3 op breakdown: zone profile of layers 0 (dense) + 3 (sparse) on the (4,4) stage-0 sub-mesh, W=2048.
cd "$(dirname "$0")"; RES="$PWD/results"
until ! pgrep -f "batch_e2w[.]sh" >/dev/null; do sleep 20; done
export TT_CACHE_PATH=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill
export GOLDEN_DIR=$TT_CACHE_PATH/golden RESULTS_DIR=$RES/profiles LOGDIR=$RES/logs
export STAGES=2 STAGE=0 LAYER_IDS=0,3 CHUNK=2048 LEVEL=2 FABRIC=1d
for h in 0 141312 548864; do for n in 2048 256; do
  echo "=== e3 h=$h n=$n $(date -Is)"
  CACHE=$h PROFILE_N_REAL=$n timeout 2700 ../models/demos/minimax_m3/scripts/run_prefill_profile.sh \
    > "$RES/logs/e3_h${h}_n${n}_r2.log" 2>&1; echo "exit=$?"
done; done
