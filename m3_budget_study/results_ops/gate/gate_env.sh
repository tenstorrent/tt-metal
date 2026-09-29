# Post-merge packed-path gate on the (2,4) stage-0 carve: layers 0-3 (3 dense + 1 sparse),
# two 2048-token segments with different histories (slot 0 at h=0, slot 1 at h=16384), real longbook tokens.
R=/home/vmelnykov/tt-metal/m3_budget_study/results_ops
export BUDGET_RESULTS=$R BUDGET_COLLECT_CSV=$R/gate/budget_runs.csv
export BUDGET_LOCK=$R/.lock BUDGET_LOCK_OWNER=vmelnykov-ops-agent
export BUDGET_TOKENS=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill/golden/longbook_56320/metadata.json
export BUDGET_STAGES=4 BUDGET_STAGE=0 BUDGET_LAYER_IDS=0,1,2,3 EXP=GATE_MERGE LAYER_SET=L0_3
export M3_FABRIC=1d EXPERT_DTYPE=bf4 M3_MOE_W_NDSHARD=1 M3_MOE_HYBRID_THRESHOLD=128 M3_MOE_DISPATCH=v1 M3_MOE_COMBINE=v1
export LOAD_TIMEOUT=1080 STALL_TIMEOUT=300
GATE_COMPOS="G=0:2048,16384:2048"
