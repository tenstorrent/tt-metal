# MLA early-Q A/B on a Galaxy 8x4: TTNN chain vs fused all_to_all_matmul, Kimi K3.1 + GLM-5.3 full sparse layers, warm.
cd "$TT_METAL_HOME"
unset TT_MESH_GRAPH_DESC_PATH
export TT_METAL_OPERATION_TIMEOUT_SECONDS=60 LOGURU_LEVEL=INFO
# Python-only fixes after the build: overlay the op package onto the installed wheel.
SITE_TTNN="$(python3 -c 'import os, ttnn; print(os.path.dirname(ttnn.__file__))' 2>/dev/null | tail -1)"
cp -a ttnn/ttnn/operations/all_to_all_matmul ttnn/ttnn/operations/all_to_all_matmul_generic ttnn/ttnn/operations/_op_contract.py "$SITE_TTNN/operations/" && echo "overlay -> $SITE_TTNN/operations"
rc=0
# Native op: persistent L1 scratch (MLA default) vs the op's per-call scratch allocation, same session.
for ps in 1 0; do
  sel=(); [ $ps = 0 ] && sel=(-k "not host_ops")
  MLA_A2AMM_IMPL=native MLA_A2AMM_PERSISTENT_SCRATCH=$ps python3 -m pytest -s --timeout=0 models/demos/deepseek_v3_d_p/tests/sparse_mla/test_early_q_fused_ab.py "${sel[@]}" 2>&1 | tee -a /data/malimpic/jobs/logs/mla_early_q_ab_glx.$SLURM_JOB_ID.out
  [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
done
mkdir -p /data/malimpic/mla-ab/glx && cp -f generated/early_q_fused_ab/results.jsonl /data/malimpic/mla-ab/glx/results.$SLURM_JOB_ID.jsonl 2>/dev/null
exit $rc
