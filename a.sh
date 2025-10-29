#!/bin/bash
#set -eo pipefail

cd $TT_METAL_HOME
export PATH=$PATH:$TT_METAL_HOME/python_env/bin

tt-smi -r
HF_MODEL="google/gemma-3-4b-it" MESH_DEVICE="N150" pytest models/demos/gemma3/tests/test_perf_vision_cross_attention_transformer.py

tt-smi -r
HF_MODEL="google/gemma-3-4b-it" MESH_DEVICE="N300" pytest models/demos/gemma3/tests/test_perf_vision_cross_attention_transformer.py

tt-smi -r
HF_MODEL="google/gemma-3-4b-it" MESH_DEVICE="T3K" pytest models/demos/gemma3/tests/test_perf_vision_cross_attention_transformer.py

# =====================

tt-smi -r
HF_MODEL="google/gemma-3-27b-it" MESH_DEVICE="N150" pytest models/demos/gemma3/tests/test_perf_vision_cross_attention_transformer.py

tt-smi -r
HF_MODEL="google/gemma-3-27b-it" MESH_DEVICE="N300" pytest models/demos/gemma3/tests/test_perf_vision_cross_attention_transformer.py

tt-smi -r
HF_MODEL="google/gemma-3-27b-it" MESH_DEVICE="T3K" pytest models/demos/gemma3/tests/test_perf_vision_cross_attention_transformer.py
