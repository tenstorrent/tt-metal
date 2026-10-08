#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass: the twelve kernels that keep one of their two defines, main's program (that define
# removed too) against the head with the other define gone, at the tests of their readings; three passes.
cd /work
export EB_NODES_FILE=/work/tests/eb_r3_ci/r9opt/ids_fin.txt EB_RUN_LIMIT=2400
for i in 1 2 3; do
  echo "##### pass $i offfin: define removed vs head"
  bash tests/eb_r3_ci/ab_off.sh tests/eb_r3_ci/r9opt/off_fin.txt -p eb_select_plugin -p eb_unskip_plugin tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_adamw.py tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_clip_grad_norm.py tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_matmul.py tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_norm.py tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py tests/ttnn/nightly/unit_tests/operations/ssm/test_ssm_repeat_and_interleave_eltwise_mul.py tests/ttnn/unit_tests/operations/sdpa/test_sdpa_decode.py tests/ttnn/unit_tests/operations/transformers/test_chunk_gated_delta_rule.py 
done
