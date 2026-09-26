# Sourced by the SP2 follow-up batches: results folder, lock guard, layer sets.
export BUDGET_RESULTS=/home/vmelnykov/tt-metal/m3_budget_study/results_sp2
export BUDGET_LOCK=$BUDGET_RESULTS/.lock BUDGET_LOCK_OWNER=vmelnykov-2e
S8P=8,9,10,11,12,13,14,15; D2=0,1,2; SP2_15=15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
sp2 () { BUDGET_STAGES=4 BUDGET_STAGE=${STAGE:-0} ./run_budget.sh; }   # (2,4) sub-mesh; stage 0 holds 0-14, stage 1 15-29
