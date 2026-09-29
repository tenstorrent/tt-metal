#!/bin/bash
# KV PCC run of the (4,2) carved from the middle-rows 4x4 sub-torus (tile 0 = galaxy rows 2-5, torus cols 0-1).
#   RUN_ID=kv_4x2t_v2_L0-6 CFG_DISPATCH=v2 CFG_COMBINE=v2 tools/run_kv_torus.sh
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="$(cd "$R/../.." && pwd)"
G=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill/golden
export PROFILE_MESH=4x2 PROFILE_STAGE=0 PROFILE_NUM_LAYERS=7 PROFILE_CHUNK=5120 PROFILE_CACHE=51200
export PREFILL_TRACE_DIR=$G/longbook_56320 PROFILE_KV_PCC=1 PROFILE_KV_DUMP="$R/bench/$RUN_ID"
export PROFILE_PARENT_MESH=4x4 PROFILE_SUBMESH=0 TT_VISIBLE_DEVICES=2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31
export CFG_DESC=$T/models/demos/deepseek_v3_d_p/experimental_descriptors/single_bh_galaxy_subtorus_xy4_graph_descriptor.textproto
export CFG_FABRIC=2d_torus_xy CFG_MOE_TOPOLOGY=ring CFG_CCL_TOPOLOGY=linear
exec "$R/tools/run_kv_4x2.sh" "$@"
