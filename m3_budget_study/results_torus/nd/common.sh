# Sourced by the nd_ batches.
export TT_METAL_HOME=/home/vmelnykov/tt-metal
export TT_CACHE_PATH=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill
export BUDGET_TOKENS=$TT_CACHE_PATH/golden/longbook_56320/metadata.json
export BUDGET_RESULTS=/home/vmelnykov/tt-metal/m3_budget_study/results_torus
export BUDGET_LOCK=$BUDGET_RESULTS/.lock BUDGET_LOCK_OWNER=vmelnykov-torus-agent
export STALL_TIMEOUT=300 LOAD_TIMEOUT=1200
S8=8,9,10,11,12,13,14,15; D2=0,1,2
RB=/home/vmelnykov/tt-metal/m3_budget_study/run_budget.sh
ND=$BUDGET_RESULTS/nd
# one run with a 20 min hard cap on top of the driver watchdog
rb () { echo "=== $RUN_ID $(date -Is)"; timeout -k 30 1200 $RB; echo "rc=$?"; }
export PATH=$ND/shim:$PATH
