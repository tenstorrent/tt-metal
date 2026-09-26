#!/bin/bash
# SP2 follow-up, Part A: (2,4) SP=2 per-layer cost model (A1 width, A2 depth, A2w wide depth, A3 packing,
# A4 zone profiles) + A5 full stages in isolation with the DRAM report (Part B sanity, Q6 memory).
cd "$(dirname "$0")"; source sp2_env.sh
export BUDGET_ANY_LAYERS=1
until ! pgrep -f "sp2_anchor|batch_anchor" >/dev/null && ! pgrep -x -f "python3 -u models/demos/minimax_m3/tests/perf/budget_sweep.py" >/dev/null; do sleep 15; done
sw () { RUN_ID=$1 EXP=$2 LAYER_SET=$3 BUDGET_LAYER_IDS=$4 BUDGET_W=$5 BUDGET_POINTS=$6 STAGE=0 sp2; }
declare -A L=([D2]=$D2 [S8P]=$S8P)
for LS in D2 S8P; do
  for W in 2048 4096 8192 10240; do sw a1_${LS,,}_w$W A1 $LS ${L[$LS]} $W 0:$W; done
  P=""; for h in 0 16384 65536 141312 309248 548864; do for n in 256 2048; do P+="$h:$n,"; done; done
  sw a2_${LS,,}_grid A2 $LS ${L[$LS]} 2048 "$P"
  for W in 4096 8192; do sw a2w_${LS,,}_w$W A2w $LS ${L[$LS]} $W 0:$W,139264:$W,548864:$W; done
done
# A3 packing (budget_packed.py), reference T1 comes from A2
for LS in S8P D2; do
  HARNESS=budget_packed.py RUN_ID=a3_${LS,,}_w4096 EXP=A3 LAYER_SET=$LS BUDGET_LAYER_IDS=${L[$LS]} BUDGET_B=2 \
    BUDGET_COMPOS="C1=0:2048,0:2048;C4=548864:2048,0:2048;C7=548864:1024,0:2048" STAGE=0 sp2
  HARNESS=budget_packed.py RUN_ID=a3_${LS,,}_w8192 EXP=A3 LAYER_SET=$LS BUDGET_LAYER_IDS=${L[$LS]} BUDGET_B=4 \
    BUDGET_COMPOS="C8=0:2048,0:2048,0:2048,0:2048;C9=548864:2048,0:2048,0:2048,0:2048" STAGE=0 sp2
done
# A4 zone profiles, layers 0 + 3 on the (2,4) stage-0 sub-mesh
( export TT_CACHE_PATH=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill
  export GOLDEN_DIR=$TT_CACHE_PATH/golden RESULTS_DIR=$BUDGET_RESULTS/profiles LOGDIR=$BUDGET_RESULTS/logs
  export STAGES=4 STAGE=0 LAYER_IDS=0,3 CHUNK=2048 LEVEL=2 FABRIC=1d SKIP_PREFIX=1 PROFILE_SKIP_COMPILE=1
  for h in 0 141312 548864; do
    grep -q "owner=$BUDGET_LOCK_OWNER" "$BUDGET_LOCK" || { echo "lock lost"; exit 3; }
    echo "=== a4 h=$h $(date -Is)"
    CACHE=$h timeout 1800 ../models/demos/minimax_m3/scripts/run_prefill_profile.sh > "$BUDGET_RESULTS/logs/a4_h${h}.log" 2>&1
    echo "exit=$?"
  done )
# A5 full pipeline stages in isolation, W=4096, with the DRAM report
P5=0:4096,139264:4096,548864:4096
for k in 0 1 2 3; do
  ids=$(seq -s, $((k * 15)) $((k * 15 + 14)))
  RUN_ID=a5_sp2_stage$k EXP=A5 LAYER_SET=SP2_ST$k BUDGET_LAYER_IDS=$ids BUDGET_W=4096 BUDGET_POINTS=$P5 BUDGET_MEM=1 STAGE=$k sp2
done
for k in 0 1; do
  ids=$(seq -s, $((k * 30)) $((k * 30 + 29)))
  RUN_ID=a5_sp4_stage$k EXP=A5 LAYER_SET=SP4_ST$k BUDGET_LAYER_IDS=$ids BUDGET_W=4096 BUDGET_POINTS=$P5 BUDGET_MEM=1 \
    BUDGET_STAGES=2 BUDGET_STAGE=$k ./run_budget.sh
done
echo "Part A done $(date -Is)"
