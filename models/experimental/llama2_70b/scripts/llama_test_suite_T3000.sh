#!/bin/bash

# Exit script if any command fails
set -e

# change directory to TT_METAL_HOME
cd $TT_METAL_HOME

# Set Llama2_70b related cached files and tests
export LLAMA_CKPT_DIR="/home/llama-data-repacked-2/llama-2-70b/"
export LLAMA_TOKENIZER_PATH="/home/llama-data/tokenizer.model"
export LLAMA_CACHE_PATH="/home/llama-data-cache/weights-cache-4"

# Run 8-chip tests on T3000s
pytest models/experimental/llama2_70b/tests/test_llama_mlp_t3000.py
pytest models/experimental/llama2_70b/tests/test_llama_attention_t3000.py
pytest models/experimental/llama2_70b/tests/test_llama_decoder_t3000.py
pytest models/experimental/llama2_70b/tests/test_llama_model_t3000.py
