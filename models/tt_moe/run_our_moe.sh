#!/bin/bash
# Run tt_moe test suite

cd /home/ntarafdar/tt-moe/tt-metal
source python_env/bin/activate
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD MESH_DEVICE=TG
export DEEPSEEK_V3_HF_MODEL=/data/deepseek/DeepSeek-R1-0528
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache_$(date +%Y%m%d_%H%M%S)_$$
mkdir -p $DEEPSEEK_V3_CACHE

echo "Running tt_moe test suite..."
pytest models/tt_moe/tests/test_moe_block.py -xvs

echo "Cache directory: $DEEPSEEK_V3_CACHE"
