#!/bin/bash
# P2: zone profiles of one (4,2) carved from the middle-rows 4x4 sub-torus (galaxy rows 2-5, cols 0-1 of the torus;
# axis 0 wraps through the chord link), layers 0-6, prose, W in {4096, 8192} x h in {0, 141312, 548864}.
#   T1 = torus fabric (2d_torus_xy) + v1 dispatch/combine on a Ring (isolates the fabric)
#   T2 = torus fabric + dispatch_fabric2d / combine_fabric2d (M3_MOE_DISPATCH=v2 M3_MOE_COMBINE=v2)
#   T3 = torus fabric + dispatch_fabric2d, v1 combine
# TP collectives stay Linear: the carve's axis 1 (2 chips) has no wrap inside the sub-mesh.
# Runs batch_p0a.sh per config; rows go to per_op_4x2_torus.csv / p2t_runs.csv.
#   nohup m3_budget_study/results_ops/batch_p2t.sh "T2 T1" > m3_budget_study/results_ops/logs/batch_p2t.out 2>&1 &
set -uo pipefail
RES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="$(cd "$RES/../.." && pwd)"
export TT_VISIBLE_DEVICES=2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/models/demos/deepseek_v3_d_p/experimental_descriptors/single_bh_galaxy_subtorus_xy4_graph_descriptor.textproto
export PROFILE_PARENT_MESH=4x4 PROFILE_SUBMESH=0 MESH=4x2
export CFG_FABRIC=2d_torus_xy CFG_MOE_TOPOLOGY=ring CFG_CCL_TOPOLOGY=linear
export PER_OP_CSV="$RES/per_op_4x2_torus.csv" RUNS_CSV="$RES/p2t_runs.csv"
export ONLY="${ONLY:-w4096_h0_prose w4096_h141312_prose w4096_h548864_prose w8192_h0_prose w8192_h141312_prose w8192_h548864_prose}"
for cfg in ${1:-T2 T1}; do
  case $cfg in
    T1) export CFG_DISPATCH=v1 CFG_COMBINE=v1 PREFIX=p2t1 ;;
    T2) export CFG_DISPATCH=v2 CFG_COMBINE=v2 PREFIX=p2t2 ;;
    T3) export CFG_DISPATCH=v2 CFG_COMBINE=v1 PREFIX=p2t3 ;;   # v2 dispatch only (KV PCC-clean)
    *) echo "unknown cfg $cfg"; exit 1 ;;
  esac
  "$RES/batch_p0a.sh" || { echo "[p2t] batch $cfg stopped rc=$?"; exit 1; }
done
env -u TT_VISIBLE_DEVICES tt-smi -glx_reset > /dev/null 2>&1
echo "[p2t] $(date '+%F %T') all done"
