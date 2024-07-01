#!/bin/bash

# Exit script if any command fails
set -e

# change directory to TT_METAL_HOME
cd $TT_METAL_HOME

# Run 8-chip tests on T3000s
pytest models/demos/t3000/llama2_70b/tests/test_llama_mlp_t3000.py
pytest models/demos/t3000/llama2_70b/tests/test_llama_attention_t3000.py
pytest models/demos/t3000/llama2_70b/tests/test_llama_decoder_t3000.py
pytest models/demos/t3000/llama2_70b/tests/test_llama_model_t3000.py
