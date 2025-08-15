import pytest
import torch
import ttnn


@pytest.mark.parametrize(
    "input_shape,split_size,dim",
    [
        # Very large YOLOv12x ultra-high resolution split operations designed to trigger OOM
        ([1, 1536, 2176, 3840], 768, 1),    # Massive C3k2 layer split
        ([1, 2304, 1088, 1920], 1152, 1),   # Large feature map split
        ([1, 3072, 544, 960], 1536, 1),     # Ultra-large channel split
        ([1, 4608, 272, 480], 2304, 1),     # Extreme channel split
        ([1, 6144, 136, 240], 3072, 1),     # Maximum channel split
        
        # Massive flattened tensor splits
        ([1, 98304, 1024], 49152, 1),       # Huge flattened attention split
        ([1, 131072, 768], 65536, 1),       # Extreme flattened feature split
        ([1, 163840, 512], 81920, 1),       # Ultra-large flattened split
        
        # Ultra-high resolution input splits
        ([1, 12, 2176, 3840], 6, 1),        # Split ultra-high-res input channels
        ([1, 24, 2176, 3840], 12, 1),       # Larger ultra-high-res split
        ([1, 48, 2176, 3840], 24, 1),       # Maximum ultra-high-res split
        
        # Memory-intensive detection head splits
        ([1, 16, 685440], 8, 1),            # Large detection head split
        ([1, 32, 342720], 16, 1),           # Medium detection head split
        ([1, 64, 171360], 32, 1),           # Heavy detection head split
    ],
)
def test_oom_split(device, input_shape, split_size, dim):
    """Test split operator OOM scenarios with very large YOLOv12x shapes"""
    torch.manual_seed(0)

    try:
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.split(ttnn_input, split_size, dim)

        # If we reach here, the operation succeeded
        print(f"✅ Split operation succeeded unexpectedly for shape {input_shape}")
        assert False, f"Expected OOM but operation succeeded for shape {input_shape}"

    except Exception as e:
        if "OutOfMemoryError" in str(e) or "out of memory" in str(e).lower():
            print(f"✅ Expected OOM for split shape {input_shape}, split_size={split_size}: {e}")
            # This is expected, test passes
            pass
        else:
            print(f"❌ Unexpected error (not OOM) for shape {input_shape}: {e}")
            raise


@pytest.mark.parametrize(
    "input_shape,split_size,dim",
    [
        # Memory estimation for YOLOv12x split operations
        ([1, 768, 2176, 3840], 384, 1),     # Large YOLOv12x split
        ([1, 1536, 1088, 1920], 768, 1),    # High-res feature split
        ([1, 3072, 544, 960], 1536, 1),     # Medium-res feature split
        ([1, 6144, 272, 480], 3072, 1),     # Lower-res feature split
    ],
)
def test_split_memory_estimation(device, input_shape, split_size, dim):
    """Estimate memory usage for split operations without running them"""
    
    # Calculate memory requirements
    input_elements = 1
    for dim_size in input_shape:
        input_elements *= dim_size
    
    # Each element is bfloat16 (2 bytes), and we need space for input + outputs
    # Split typically creates 2 or more outputs, estimate as 3x memory usage
    estimated_memory_bytes = input_elements * 2 * 3  # input + 2 outputs
    estimated_memory_mb = estimated_memory_bytes / (1024 * 1024)
    
    print(f"📊 Shape {input_shape}:")
    print(f"   Input elements: {input_elements:,}")
    print(f"   Estimated memory: {estimated_memory_mb:.2f} MB")
    print(f"   Split size: {split_size}, dim: {dim}")
    
    # L1 memory is typically limited (e.g., 1MB), flag if likely to exceed
    if estimated_memory_mb > 1.0:
        print(f"   ⚠️  Likely to exceed L1 memory limit (>1MB)")
    else:
        print(f"   ✅ Should fit in L1 memory")
    
    # This test always passes, it's just for memory analysis
    assert True
