#!/bin/bash
# SP=8 (whole 8x4, config E3) per-layer cost model runs. Usage: batch_sp8.sh [phase ...]  (W D Dw P ST; default all)
S=/home/vmelnykov/tt-metal/m3_budget_study; DRV=$S/run_budget.sh
export TT_METAL_HOME=/home/vmelnykov/tt-metal
export M3_FABRIC=2d_torus_xy TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
export M3_CCL_TOPOLOGY=ring M3_MOE_TOPOLOGY=ring M3_MOE_DISPATCH=v2 M3_MOE_COMBINE=v2
export TT_CACHE_PATH=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill
export BUDGET_TOKENS=$TT_CACHE_PATH/golden/longbook_56320/metadata.json
export BUDGET_STAGES=1 BUDGET_RESULTS=$S/results_torus
export BUDGET_LOCK=$BUDGET_RESULTS/.lock BUDGET_LOCK_OWNER=vmelnykov-torus-agent
RES=$BUDGET_RESULTS; BLOG=$RES/sp8/batch.log
NOTES="E3 M3_FABRIC=2d_torus_xy M3_CCL_TOPOLOGY=ring M3_MOE_TOPOLOGY=ring M3_MOE_DISPATCH=v2 M3_MOE_COMBINE=v2 tokens=longbook_56320"
declare -A L=([D8]=0,1,2 [S8]=8,9,10,11,12,13,14,15 [S5]=3,4,5,6,7
  [ST0]=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 [ST1]=15,16,17,18,19,20,21,22,23,24,25,26,27,28,29)
HANGS=0
run () {  # RUN_ID EXP LAYER_SET VAR=val ...
  local id=$1 exp=$2 ls=$3; shift 3
  [ -e $RES/logs/$id.log ] && { echo "$(date +%T) skip $id (log exists)" >> $BLOG; return; }
  echo "$(date +%T) start $id $*" >> $BLOG
  env RUN_ID=$id EXP=$exp LAYER_SET=$ls BUDGET_LAYER_IDS=${L[$ls]} "$@" $DRV > $RES/sp8/drv_$id.out 2>&1 &
  local dpid=$! t0=$(date +%s) to=""
  until [ -e $RES/logs/$id.env ] || ! kill -0 $dpid 2>/dev/null; do sleep 1; done
  echo "NOTES=$NOTES" >> $RES/logs/$id.env
  while kill -0 $dpid 2>/dev/null; do
    sleep 10
    if [ -z "$to" ] && [ $(( $(date +%s) - t0 )) -gt 1300 ]; then  # 20 min of process time + reset
      local py; py=$(pgrep -P $dpid -x python3)
      [ -n "$py" ] && { kill -TERM $py; sleep 15; kill -KILL $py 2>/dev/null; }; to=1
    fi
  done
  wait $dpid
  local st; st=$(grep -o "^STATUS=[A-Z_]*" $RES/logs/$id.log | tail -1)
  [ -n "$to" ] && st="TIMEOUT20($st)"
  echo "$(date +%T) end $id $st $(tail -1 $RES/sp8/drv_$id.out)" >> $BLOG
  case $st in *HANG*|*TIMEOUT*) HANGS=$((HANGS+1));; *) HANGS=0;; esac
  [ $HANGS -ge 3 ] && { echo "$(date +%T) 3 hangs in a row, stopping" >> $BLOG; exit 4; }
}
phases=${*:-W D Dw P ST}; case "$phases" in Dx*|F) phases="";; esac
for ph in $phases; do case $ph in
W)  for LS in D8 S8; do for W in 2048 4096 8192 16384; do
      run s8_p8w_${LS,,}_w$W P8W $LS BUDGET_W=$W BUDGET_POINTS=0:$W; done; done
    for W in 4096 8192; do run s8_p8w_s5_w$W P8W S5 BUDGET_W=$W BUDGET_POINTS=0:$W; done ;;
D)  PTS=""; for h in 0 16384 65536 139264 311296 548864; do for n in 4096 512; do PTS+="$h:$n,"; done; done
    for LS in D8 S8; do run s8_p8d_${LS,,}_w4096 P8D $LS BUDGET_W=4096 BUDGET_POINTS=${PTS%,}; done ;;
Dw) for LS in D8 S8; do
      run s8_p8dw_${LS,,}_w8192 P8Dw $LS BUDGET_W=8192 BUDGET_POINTS=0:8192,139264:8192,548864:8192
      run s8_p8dw_${LS,,}_w16384 P8Dw $LS BUDGET_W=16384 BUDGET_POINTS=0:16384,147456:16384,540672:16384; done ;;
P)  for LS in S8 D8; do
      run s8_p8p_${LS,,}_b2 P8P $LS HARNESS=budget_packed.py BUDGET_B=2 "BUDGET_COMPOS=C1=0:2048,0:2048;C4=548864:2048,0:2048"
      run s8_p8p_${LS,,}_b4 P8P $LS HARNESS=budget_packed.py BUDGET_B=4 "BUDGET_COMPOS=C8=0:2048,0:2048,0:2048,0:2048;C9=548864:2048,0:2048,0:2048,0:2048"; done ;;
ST) for LS in ST0 ST1; do
      run s8_p8st_${LS,,}_w8192 P8ST $LS BUDGET_W=8192 BUDGET_POINTS=0:8192,139264:8192,548864:8192 BUDGET_MEM=1; done ;;
esac; done
echo "$(date +%T) batch done ($phases)" >> $BLOG
# Dx: where does the sparse W=8192 depth step start (0 -> 139264 costs +39 ms, W=4096/16384 do not step)?
[ "${1:-}" = Dx ] && run s8_p8dx_s8_w8192 P8Dx S8 BUDGET_W=8192 BUDGET_POINTS=0:8192,8192:8192,16384:8192,32768:8192,65536:8192,139264:8192
# Dx2: same points at the P8Dw capacity (557056): is the step capacity-driven?
[ "${1:-}" = Dx2 ] && run s8_p8dx2_s8_w8192_cap557056 P8Dx S8 BUDGET_W=8192 BUDGET_CAPACITY=557056 BUDGET_POINTS=0:8192,8192:8192,65536:8192,139264:8192
# Dx3: exact repeat of s8_p8dw_s8_w8192 (reproducibility of its 139264 point)
[ "${1:-}" = Dx3 ] && run s8_p8dw_s8_w8192_r2 P8Dxr S8 BUDGET_W=8192 BUDGET_POINTS=0:8192,139264:8192,548864:8192
# F: "fast-mode" reruns — a repeated first deep point (8192:8192) before the deeper points
if [ "${1:-}" = F ]; then
  run s8_p8dwf_s8_w8192 P8Dwf S8 BUDGET_W=8192 BUDGET_POINTS=0:8192,8192:8192,139264:8192,548864:8192
  run s8_p8stf_st1_w8192 P8STf ST1 BUDGET_W=8192 BUDGET_POINTS=0:8192,8192:8192,139264:8192,548864:8192 BUDGET_MEM=1
fi
