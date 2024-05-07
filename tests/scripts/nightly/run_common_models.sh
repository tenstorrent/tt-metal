#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

echo "Running common models for archs"

env pytest models/demos/metal_BERT_large_11/tests/test_bert_batch_dram.py
env pytest models/demos/metal_BERT_large_11/tests/test_demo.py

env pytest models/demos/falcon7b/tests/ci/test_falcon_end_to_end_prefill.py

env pytest models/demos/ttnn_falcon7b/tests/test_falcon_mlp.py
env pytest models/demos/ttnn_falcon7b/tests/test_falcon_rotary_embedding.py
env pytest models/demos/ttnn_falcon7b/tests/test_falcon_attention.py
env pytest models/demos/ttnn_falcon7b/tests/test_falcon_decoder.py
