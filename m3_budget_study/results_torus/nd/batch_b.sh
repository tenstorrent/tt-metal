#!/bin/bash
# Block B: SP=4. usage: batch_b.sh "<CFG:W> ..." with CFG in M4A M4R M4E3, W in 4096 8192
source /home/vmelnykov/tt-metal/m3_budget_study/results_torus/nd/common.sh
export BUDGET_STAGES=2 BUDGET_STAGE=0 EXP=NDB LAYER_SET=S8 BUDGET_LAYER_IDS=$S8 M3_MOE_W_NDSHARD=1
unset M3_MOE_HYBRID_THRESHOLD
P4=0:4096,139264:4096,548864:4096; P8=0:8192,8192:8192,139264:8192,548864:8192
cfg () { unset TT_VISIBLE_DEVICES BUDGET_MESH TT_MESH_GRAPH_DESC_PATH M3_CCL_TOPOLOGY M3_MOE_TOPOLOGY M3_MOE_DISPATCH M3_MOE_COMBINE
  case $1 in
  M4A) export M3_FABRIC=1d;;
  M4R|M4E3) export TT_VISIBLE_DEVICES=2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31 BUDGET_MESH=4x4 M3_FABRIC=2d_torus_xy \
      TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/models/demos/deepseek_v3_d_p/experimental_descriptors/single_bh_galaxy_subtorus_xy4_graph_descriptor.textproto \
      M3_CCL_TOPOLOGY=ring M3_MOE_TOPOLOGY=ring
    [ $1 = M4E3 ] && export M3_MOE_DISPATCH=v2 M3_MOE_COMBINE=v2;;
  esac
  export NOTES="$1 sp4 M3_FABRIC=$M3_FABRIC ${BUDGET_MESH:+mesh=$BUDGET_MESH subtorus} M3_CCL_TOPOLOGY=${M3_CCL_TOPOLOGY:-linear} M3_MOE_TOPOLOGY=${M3_MOE_TOPOLOGY:-linear} M3_MOE_DISPATCH=${M3_MOE_DISPATCH:-v1} M3_MOE_COMBINE=${M3_MOE_COMBINE:-v1} M3_MOE_W_NDSHARD=1 tokens=longbook_56320"; }
for cw in $1; do C=${cw%:*}; W=${cw#*:}; cfg $C; [ $W = 4096 ] && P=$P4 || P=$P8; P=${POINTS_OVERRIDE:-$P}
  RUN_ID=nd_b_${C,,}_s8_w$W${SUFFIX:-} BUDGET_W=$W BUDGET_POINTS=$P rb
done
tt-smi -glx_reset > /dev/null 2>&1; echo "=== batch_b done $(date -Is)"
