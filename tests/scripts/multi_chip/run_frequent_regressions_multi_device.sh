#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

if [[ -z "$ARCH_NAME" ]]; then
  echo "Must provide ARCH_NAME in environment" 1>&2
  exit 1
fi

cd $TT_METAL_HOME
export PYTHONPATH=$TT_METAL_HOME

pytest tests/ttnn/unit_tests/test_multi_device.py

pytest tests/tt_metal/microbenchmarks/ethernet/test_ethernet_bidirectional_bandwidth_microbenchmark.py
pytest tests/tt_metal/microbenchmarks/ethernet/test_ethernet_ring_latency_microbenchmark.py

# Llama2_70b related cached files and tests (the test should parse env variables similar to these)
export LLAMA_CKPT_DIR=/mnt/MLPerf/tt_dnn-models/llama-2/llama-2-70b-repacked/
export LLAMA_TOKENIZER_PATH=/mnt/MLPerf/tt_dnn-models/llama-2/tokenizer.model
export LLAMA_CACHE_PATH=/mnt/MLPerf/tt_dnn-models/llama-2/llama-data-cache/weights-cache-2

pytest models/demos/t3000/llama2_70b/tests/test_llama_mlp_t3000.py
pytest models/demos/t3000/llama2_70b/tests/test_llama_attention_t3000.py
pytest models/demos/t3000/llama2_70b/tests/test_llama_decoder_t3000.py
pytest models/demos/t3000/llama2_70b/tests/test_llama_model_t3000.py
