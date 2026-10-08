#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass, merged head: the test_eb_r3_ops.py cases by -k (their node ids did not match in r9oh):
# bge's balanced layernorm with its standard define removed against kept, and the HF rotary standard, the large layernorm
# and large softmax broadcast defines removed against kept; three passes.
cd /work
export EB_RUN_LIMIT=2400
grep balanced_layernorm tests/eb_r3_ci/r9opt/off_h.txt > /tmp/off_bge.txt
grep -E "rotary_embedding_hf.cpp|layernorm_large_tensor.cpp|softmax_large_tensor.cpp" tests/eb_r3_ci/r9opt/off_w.txt > /tmp/off_ops.txt
cat /tmp/off_bge.txt /tmp/off_ops.txt
for i in 1 2 3; do
  echo "##### pass $i offhbge: define removed vs head"
  EB_K_EXPR="test_bge_balanced_layernorm" bash tests/eb_r3_ci/ab_off.sh /tmp/off_bge.txt -p eb_k_plugin tests/eb_r3_ci/test_eb_r3_ops.py
  echo "##### pass $i offwops: define removed vs head"
  EB_K_EXPR="test_rotary_hf_prefill or test_layer_norm_large or test_softmax_large" bash tests/eb_r3_ci/ab_off.sh /tmp/off_ops.txt -p eb_k_plugin tests/eb_r3_ci/test_eb_r3_ops.py
done
