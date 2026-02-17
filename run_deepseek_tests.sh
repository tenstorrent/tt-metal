#!/bin/bash
# Run DeepSeek MoE tests for both decode and prefill modes

# Setup environment
cd /home/ntarafdar/tt-moe/tt-metal
source python_env/bin/activate
export PYTHONPATH=$PWD TT_METAL_HOME=$PWD MESH_DEVICE=TG
export DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52
export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache

# Function to run a specific test mode
run_test() {
    local mode=$1
    echo "============================================="
    echo "Running $mode mode test..."
    echo "============================================="

    if [[ "$mode" == "decode" ]]; then
        pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference[mode_decode_seq_1] -xvs
    elif [[ "$mode" == "prefill" ]]; then
        pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference[mode_prefill_seq_128] -xvs
    elif [[ "$mode" == "both" ]]; then
        pytest models/tt-moe/tests/test_deepseek_moe_block.py::test_deepseek_moe_against_reference -xvs
    else
        echo "Invalid mode. Use: decode, prefill, or both"
        exit 1
    fi
}

# Parse command line argument
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [decode|prefill|both]"
    echo "  decode  - Run decode mode test (seq_len=1, batch=32)"
    echo "  prefill - Run prefill mode test (seq_len=128, batch=1)"
    echo "  both    - Run both test modes"
    exit 1
fi

# Run the test
run_test $1

echo ""
echo "Test completed!"
