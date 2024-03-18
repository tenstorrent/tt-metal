#!/bin/bash

# Exit script if any command fails
set -e

# change directory to TT_METAL_HOME
cd $TT_METAL_HOME

# Run all emulated tests
pytest -svv models/demos/llama2_70b/tests/test_llama_mlp.py::test_LlamaMLP_inference[BFLOAT16-DRAM-0.9999-decode-8chip-emulated]

pytest -svv models/demos/llama2_70b/tests/test_llama_attention.py::test_LlamaAttention_inference[BFLOAT16-DRAM-0.9997-decode-8chip-emulated]

pytest -svv models/demos/llama2_70b/tests/test_llama_decoder.py::test_LlamaDecoder_inference[BFLOAT16-DRAM-0.999-decode-8chip-emulated]

pytest -svv models/demos/llama2_70b/tests/test_llama_model.py::test_LlamaModel_inference[BFLOAT16-DRAM-decode-8chip-emulated-0.999-1]

pytest -svv models/demos/llama2_70b/tests/test_llama_model.py::test_LlamaModel_inference[BFLOAT16-DRAM-decode-8chip-emulated-0.998-2]

pytest -svv models/demos/llama2_70b/tests/test_llama_mlp.py::test_LlamaMLP_inference[BFLOAT16-DRAM-0.9999-decode-32chip-emulated]

pytest -svv models/demos/llama2_70b/tests/test_llama_attention.py::test_LlamaAttention_inference[BFLOAT16-DRAM-0.9997-decode-32chip-emulated]

pytest -svv models/demos/llama2_70b/tests/test_llama_decoder.py::test_LlamaDecoder_inference[BFLOAT16-DRAM-0.999-decode-32chip-emulated]
