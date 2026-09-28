#!/bin/bash
# Block A: SP=2 (2,4) stage 0, 1d, v1/v1; N0 / N1 / N1H.
source /home/vmelnykov/tt-metal/m3_budget_study/results_torus/nd/common.sh
export BUDGET_STAGES=4 BUDGET_STAGE=0 BUDGET_ANY_LAYERS=1 M3_FABRIC=1d EXP=NDA
unset TT_VISIBLE_DEVICES BUDGET_MESH M3_CCL_TOPOLOGY M3_MOE_TOPOLOGY M3_MOE_DISPATCH M3_MOE_COMBINE
cfg () { case $1 in
  N0) export M3_MOE_W_NDSHARD=0; unset M3_MOE_HYBRID_THRESHOLD;;
  N1) export M3_MOE_W_NDSHARD=1; unset M3_MOE_HYBRID_THRESHOLD;;
  N1H) export M3_MOE_W_NDSHARD=1 M3_MOE_HYBRID_THRESHOLD=128;; esac
  export NOTES="$1 sp2 1d v1 M3_MOE_W_NDSHARD=$M3_MOE_W_NDSHARD M3_MOE_HYBRID_THRESHOLD=${M3_MOE_HYBRID_THRESHOLD:-0} tokens=longbook_56320"; }
for C in N0 N1 N1H; do cfg $C
  RUN_ID=nd_a_${C,,}_s8p_w2048 LAYER_SET=S8P BUDGET_LAYER_IDS=$S8 BUDGET_W=2048 BUDGET_POINTS=0:2048,16384:2048,141312:2048,548864:2048 rb
  RUN_ID=nd_a_${C,,}_s8p_w4096 LAYER_SET=S8P BUDGET_LAYER_IDS=$S8 BUDGET_W=4096 BUDGET_POINTS=0:4096,139264:4096,548864:4096 rb
done
for C in N0 N1; do cfg $C
  RUN_ID=nd_a_${C,,}_d2_w4096 LAYER_SET=D2 BUDGET_LAYER_IDS=$D2 BUDGET_W=4096 BUDGET_POINTS=0:4096 rb
done
# zone profiles
export GOLDEN_DIR=$TT_CACHE_PATH/golden SRC_TRACE=$TT_CACHE_PATH/golden/longbook_56320/metadata.json
export RESULTS_DIR=$ND/profiles LOGDIR=$BUDGET_RESULTS/logs STAGES=4 STAGE=0 LAYER_IDS=0,3 CHUNK=2048 CACHE=0 PROFILE_SKIP_COMPILE=1 LEVEL=2
mkdir -p $RESULTS_DIR
for C in N0 N1 N1H; do cfg $C
  grep -q "owner=$BUDGET_LOCK_OWNER" "$BUDGET_LOCK" || { echo "lock lost"; exit 3; }
  echo "=== prof $C $(date -Is)"; tt-smi -glx_reset > /dev/null 2>&1
  FABRIC=nd_$C timeout -k 30 1200 $TT_METAL_HOME/models/demos/minimax_m3/scripts/run_prefill_profile.sh > $BUDGET_RESULTS/logs/nd_prof_a_$C.out 2>&1
  echo "rc=$?"
done
tt-smi -glx_reset > /dev/null 2>&1
echo "=== batch_a done $(date -Is)"
