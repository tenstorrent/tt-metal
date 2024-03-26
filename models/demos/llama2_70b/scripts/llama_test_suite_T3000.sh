#!/bin/bash

# Exit script if any command fails
set -e

# change directory to TT_METAL_HOME
cd $TT_METAL_HOME

# Run 8-chip tests on T3000s
pytest models/demos/llama2_70b/tests/test_llama_mlp.py::test_LlamaMLP_inference[decode-8chip-T3000]
pytest models/demos/llama2_70b/tests/test_llama_attention.py::test_LlamaAttention_inference[decode-8chip-T3000]
pytest models/demos/llama2_70b/tests/test_llama_decoder.py::test_LlamaDecoder_inference[decode-8chip-T3000]
pytest models/demos/llama2_70b/tests/test_llama_model.py::test_LlamaModel_inference[decode-8chip-T3000-1L]
