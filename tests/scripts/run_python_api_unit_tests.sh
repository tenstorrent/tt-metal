#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

pytest $TT_METAL_HOME/tests/python_api_testing/unit_testing/ -vvv
pytest $TT_METAL_HOME/tests/python_api_testing/sweep_tests/pytests/ -vvv

# For now, adding tests with fast dispatch and non-32B divisible page sizes here. Python/models people,
# you can move to where you'd like.
env TT_METAL_DEVICE_DISPATCH_MODE=1 python tests/python_api_testing/models/mnist/test_mnist.py
env TT_METAL_DEVICE_DISPATCH_MODE=1 pytest tests/python_api_testing/sweep_tests/pytests/tt_dnn/test_pad.py
env TT_METAL_DEVICE_DISPATCH_MODE=1 pytest tests/python_api_testing/sweep_tests/pytests/tt_dnn/test_unpad.py

# Tests for tensors in L1
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large*matmul* -k in0_L1-in1_L1-bias_L1-out_L1
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large*bmm* -k in0_L1-in1_L1-out_L1
# Tests for mixed precision (sweeps combos of bfp8_b/bfloat16 dtypes for fused_qkv_bias and ff1_bias_gelu matmul and pre_softmax_bmm)
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large_matmuls_and_bmms_with_mixed_precision.py::test_bert_large_matmul -k "fused_qkv_bias and batch_9 and L1"
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large_matmuls_and_bmms_with_mixed_precision.py::test_bert_large_matmul -k "ff1_bias_gelu and batch_9 and DRAM"
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large_matmuls_and_bmms_with_mixed_precision.py::test_bert_large_bmm -k "pre_softmax_bmm and batch_9"

# TODO: Remove split fused and create qkv heads tests if we delete these TMs?
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large_create_qkv_heads_from_fused_qkv.py -k "in0_L1-out_L1 and batch_9"
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large_split_fused_qkv.py -k "in0_L1-out_L1 and batch_9"
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large_create_qkv_heads.py -k "in0_L1-out_L1 and batch_9"
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large_concat_heads.py -k "in0_L1-out_L1 and batch_9"

# Test program cache
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/ -k program_cache

# Fused ops unit tests
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/fused_ops/test_bert_large_fused_ln.py -k "in0_L1-out_L1 and batch_9"
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/fused_ops/test_bert_large_fused_softmax.py -k "in0_L1 and batch_9"

# Resnet18 tests with conv on cpu and with conv on device
pytest $TT_METAL_HOME/tests/python_api_testing/models/resnet/test_resnet18.py

# Conv unit test with fast dispatch
env TT_METAL_DEVICE_DISPATCH_MODE=1 pytest -svv tests/python_api_testing/unit_testing/test_conv_as_large_matmul.py::test_run_conv_as_large_matmul[32-3-5-5-1-1-1-1-0-0-False]
