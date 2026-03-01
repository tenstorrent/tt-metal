#!/bin/bash
# Run our MoE implementation only (no comparison)

cd /home/ntarafdar/tt-moe/tt-metal
source python_env/bin/activate
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD MESH_DEVICE=TG
export DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache

echo "Running our MoE implementation only..."
pytest models/tt-moe/tests/test_our_moe_only.py::test_our_moe_only -xvs
