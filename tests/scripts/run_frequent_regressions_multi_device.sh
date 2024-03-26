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

# Llama2_70b related cached files and tests (the test should parse env variables similar to these)
export LLAMA_CKPT_DIR=/opt/cache/llama2_70b_cache/llama-data-repacked-2/llama-2-70b/
export LLAMA_TOKENIZER_PATH=/opt/cache/llama2_70b_cache/llama-data/tokenizer.model
export LLAMA_CACHE_PATH=/opt/cache/llama2_70b_cache/llama-data-cache/weights-cache-2

pytest models/demos/llama2_70b/tests/test_llama_mlp.py::test_LlamaMLP_inference[decode-8chip-T3000]
pytest models/demos/llama2_70b/tests/test_llama_attention.py::test_LlamaAttention_inference[decode-8chip-T3000]
pytest models/demos/llama2_70b/tests/test_llama_decoder.py::test_LlamaDecoder_inference[decode-8chip-T3000]
pytest models/demos/llama2_70b/tests/test_llama_model.py::test_LlamaModel_inference[decode-8chip-T3000-1L]
