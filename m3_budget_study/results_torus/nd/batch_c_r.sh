#!/bin/bash
# Block C: SP=8 whole 8x4, E3, M3_MOE_W_NDSHARD 0 vs 1.
source /home/vmelnykov/tt-metal/m3_budget_study/results_torus/nd/common.sh
unset TT_VISIBLE_DEVICES BUDGET_MESH M3_MOE_HYBRID_THRESHOLD
export BUDGET_STAGES=1 EXP=NDC LAYER_SET=S8 BUDGET_LAYER_IDS=$S8 M3_FABRIC=2d_torus_xy M3_CCL_TOPOLOGY=ring M3_MOE_TOPOLOGY=ring \
  M3_MOE_DISPATCH=v2 M3_MOE_COMBINE=v2 TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
for n in ${NLIST:-0 1}; do export M3_MOE_W_NDSHARD=$n NOTES="E3 sp8 M3_FABRIC=2d_torus_xy ring/ring v2/v2 M3_MOE_W_NDSHARD=$n tokens=longbook_56320"
  RUN_ID=nd_c_e3_n${n}_s8_w4096${SUFFIX:-} BUDGET_W=4096 BUDGET_POINTS=0:4096,548864:4096 rb
done
tt-smi -glx_reset > /dev/null 2>&1; echo "=== batch_c done $(date -Is)"
