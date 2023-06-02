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
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large*matmul* -k in0_L1-in1_L1-out_L1
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large*bmm* -k in0_L1-in1_L1-out_L1
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large_split_fused_qkv.py -k in0_L1-out_L1
pytest $TT_METAL_HOME/tests/python_api_testing/models/bert_large_performant/unit_tests/test_bert_large_create_qkv_heads.py -k in0_L1-out_L1
