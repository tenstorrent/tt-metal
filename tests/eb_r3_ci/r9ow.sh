#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass, merged head: the kept defines whose reading was one pass under 0.5 % or taken over a
# define since reverted, each removed against kept, three passes.
cd /work
export EB_NODES_FILE=/work/tests/eb_r3_ci/r9opt/ids_w.txt EB_RUN_LIMIT=2400
for i in 1 2 3; do
  echo "##### pass $i offw: define removed vs head"
  bash tests/eb_r3_ci/ab_off.sh tests/eb_r3_ci/r9opt/off_w.txt -p eb_select_plugin -p eb_unskip_plugin tests/eb_r3_ci/test_eb_r3_ops.py tests/tt_eager/python_api_testing/unit_testing/misc/test_rotary_embedding_hf.py tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_mhc_split_sinkhorn.py tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_moe_hash_gate.py tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_router_gpt.py tests/ttnn/nightly/unit_tests/operations/ssm/test_ssm_prefix_scan.py tests/ttnn/nightly/unit_tests/operations/transformers/test_distributed_fused_rmsnorm.py 
done
