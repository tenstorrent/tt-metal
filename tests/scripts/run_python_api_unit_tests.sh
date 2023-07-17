#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

pytest $TT_METAL_HOME/tests/python_api_testing/unit_testing/ -vvv
pytest $TT_METAL_HOME/tests/python_api_testing/sweep_tests/pytests/ -vvv

# Tests for tensors in L1
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large*matmul* -k in0_L1-in1_L1-bias_L1-out_L1
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large*bmm* -k in0_L1-in1_L1-out_L1
# Tests for mixed precision (sweeps all combos of bfp8_b/bfloat16 dtypes for ff1 matmul (with bias and gelu) and pre_softmax_bmm)
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large_matmuls_and_bmms_with_mixed_precision.py::test_bert_large_matmul -k ff1_bias_gelu
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large_matmuls_and_bmms_with_mixed_precision.py::test_bert_large_bmm -k pre_softmax_bmm

# TODO: Remove split fused and create qkv heads tests if we delete these TMs?
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large_create_qkv_heads_from_fused_qkv.py -k in0_L1-out_L1
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large_split_fused_qkv.py -k in0_L1-out_L1
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large_create_qkv_heads.py -k in0_L1-out_L1
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large_concat_heads.py -k in0_L1-out_L1

# Test program cache
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/ -k program_cache

# Fused ops unit tests
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/fused_ops/test_bert_large_fused_ln.py -k in0_L1-out_L1
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/fused_ops/test_bert_large_fused_softmax.py -k in0_L1
