#!/bin/bash
# E3 at depth: zoned attention cost with a zeroed cache and no bucket sweep (small tracy capture).
cd "$(dirname "$0")"; RES="$PWD/results"
until ! pgrep -f "batch_layers[.]sh" >/dev/null; do sleep 20; done
export TT_CACHE_PATH=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill
export GOLDEN_DIR=$TT_CACHE_PATH/golden RESULTS_DIR=$RES/profiles LOGDIR=$RES/logs
export STAGES=2 STAGE=0 LAYER_IDS=0,3 CHUNK=2048 LEVEL=2 FABRIC=1d SKIP_PREFIX=1 PROFILE_SKIP_COMPILE=1
for h in 0 141312 548864; do for n in 2048 256; do
  echo "=== e3b h=$h n=$n $(date -Is)"
  CACHE=$h PROFILE_N_REAL=$n timeout 1800 ../models/demos/minimax_m3/scripts/run_prefill_profile.sh \
    > "$RES/logs/e3b_h${h}_n${n}.log" 2>&1; echo "exit=$?"
  du -sh ../generated/profiler/.logs 2>/dev/null
done; done
