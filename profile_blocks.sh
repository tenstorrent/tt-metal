#!/bin/bash
# Profile OpenVLA block-by-block for optimization

set -e

# Export environment variables BEFORE profiling
export HF_MODEL="meta-llama/Llama-2-7b-hf"
export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_PROFILER_SYNC=1              # Synchronous profiling to avoid buffer drops
export ENABLE_TRACY=1
# Increase profiler buffer size to prevent overflow (default ~16MB). Use 512MB for full model.
export TT_METAL_DEVICE_PROFILER_BUFFER_SIZE=536870912

echo "========================================"
echo "OpenVLA Block-by-Block Profiling"
echo "Environment:"
echo "  HF_MODEL=$HF_MODEL"
echo "  DEVICE_PROFILER=$TT_METAL_DEVICE_PROFILER"
echo "========================================"

# Function to profile a single block
profile_block() {
    local block_num=$1
    local block_name=$2
    local test_name=$3

    echo ""
    echo ">>> Profiling Block $block_num: $block_name"
    echo "----------------------------------------"

    # Environment is already exported, just run pytest
    python3 tools/tracy/profile_this.py \
        -n "block${block_num}_${block_name}" \
        -c "pytest test_openvla_blocks.py::$test_name -v -s"

    echo "✓ Block $block_num complete!"
}

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 [all|1|2|3|4|5|full]"
    echo ""
    echo "Blocks:"
    echo "  1    - DinoV2 Vision Backbone"
    echo "  2    - SigLIP Vision Backbone"
    echo "  3    - Projector"
    echo "  4    - LLM Prefill"
    echo "  5    - LLM Decode Single Step"
    echo "  full - Full OpenVLA Model (32 layers, 7 actions)"
    echo "  all  - Profile all blocks sequentially"
    exit 1
fi

case "$1" in
    all)
        echo "Profiling ALL blocks (this will take ~10 minutes)..."
        profile_block 1 "dinov2" "test_block1_dinov2_only"
        profile_block 2 "siglip" "test_block2_siglip_only"
        profile_block 3 "projector" "test_block3_projector_only"
        profile_block 4 "llm_prefill" "test_block4_llm_prefill_only"
        profile_block 5 "llm_decode" "test_block5_llm_decode_only"
        ;;
    1)
        profile_block 1 "dinov2" "test_block1_dinov2_only"
        ;;
    2)
        profile_block 2 "siglip" "test_block2_siglip_only"
        ;;
    3)
        profile_block 3 "projector" "test_block3_projector_only"
        ;;
    4)
        profile_block 4 "llm_prefill" "test_block4_llm_prefill_only"
        ;;
    5)
        profile_block 5 "llm_decode" "test_block5_llm_decode_only"
        ;;
    full)
        echo "Profiling Full OpenVLA Model (32 layers, 7 actions)..."
        python3 tools/tracy/profile_this.py \
            -n "full_openvla_model" \
            -c "pytest models/tt_transformers/tt/multimodal/open_vla.py::test_openvla_model -v -s"
        echo "✓ Full model profiling complete!"
        ;;
    *)
        echo "Invalid option: $1"
        echo "Use: all, 1, 2, 3, 4, 5, or full"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Profiling Complete!"
echo "========================================"
echo "Results in: generated/profiler/block*/"
echo ""
echo "To view results:"
echo "  ls -lh generated/profiler/"
