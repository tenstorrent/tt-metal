#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

source build/python_env/bin/activate
export PYTHONPATH=$TT_METAL_HOME
# Needed for tests in models
python -m pip install -r tests/python_api_testing/requirements.txt

pytest $TT_METAL_HOME/tests/python_api_testing/unit_testing/ -vvv

# Tests for tensors in L1
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large*matmul* -k in0_L1-in1_L1-bias_L1-out_L1
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large*bmm* -k in0_L1-in1_L1-out_L1
pytest tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large_create_qkv_heads_from_fused_qkv.py -k in0_L1-out_L1
# TODO: Remove split fused and create qkv heads tests if we delete these TMs?
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large_split_fused_qkv.py -k in0_L1-out_L1
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large_create_qkv_heads.py -k in0_L1-out_L1
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large_concat_heads.py -k in0_L1-out_L1

# Test program cache
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/ -k program_cache

# Fused ops unit tests
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/fused_ops/test_bert_large_fused_ln.py -k in0_L1-out_L1
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/fused_ops/test_bert_large_fused_softmax.py -k in0_L1

# BERT large via new enqueue APIs. I know this is not a unit test, but I would like to avoid BERT large breaking, so this
# is a safe place to put it for the time being. Need to run these as separate tests to avoid segfault (TODO(agrebenisan): Investigate why)
env TT_METAL_DEVICE_DISPATCH_MODE=1 pytest -svv tests/python_api_testing/models/metal_BERT_large_15/test_bert_batch_dram.py::test_bert_batch_dram[DRAM]
env TT_METAL_DEVICE_DISPATCH_MODE=1 pytest -svv tests/python_api_testing/models/metal_BERT_large_15/test_bert_batch_dram.py::test_bert_batch_dram_with_program_cache[DRAM]
