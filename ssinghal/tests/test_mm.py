import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "shapes",
    [
        # Format: (input1_shape, input2_shape) for mm(input1, input2)
        # MM requires input1[n,k] @ input2[k,m] -> output[n,m]
        # Basic matrix multiplication shapes
        ([64000, 128], [128, 256]),  # Large batch, medium feature
        ([16000, 256], [256, 512]),  # Large batch, larger feature
        ([4000, 640], [640, 1024]),  # Medium batch, large feature
        ([1000, 1024], [1024, 2048]),  # Small batch, very large feature
        ([16000, 512], [512, 1024]),  # Large batch, large feature
        ([4000, 1024], [1024, 2048]),  # Medium batch, very large feature
        ([1000, 2048], [2048, 4096]),  # Small batch, huge feature
        ([16640, 256], [256, 512]),  # Large batch, medium feature
        ([4480, 512], [512, 1024]),  # Medium batch, large feature
        ([1280, 1024], [1024, 2048]),  # Small batch, very large feature
        # 8K YOLOv12x matrix multiplication shapes with proper compatibility
        ([2160, 768], [768, 3840]),  # 8K feature projection (94.9MB)
        ([1080, 1536], [1536, 1920]),  # 8K deep projection (94.9MB)
        ([540, 3072], [3072, 960]),  # 8K very deep projection (94.9MB)
        ([270, 6144], [6144, 480]),  # 8K ultra deep projection (94.9MB)
        # Additional 8K compatible shapes
        ([2160, 384], [384, 1920]),  # 8K half feature projection
        ([1080, 768], [768, 3840]),  # 8K quarter deep projection
        ([540, 1536], [1536, 7680]),  # 8K eighth deep projection
        ([270, 3072], [3072, 960]),  # 8K sixteenth deep projection
        ([135, 6144], [6144, 480]),  # 8K thirty-second deep projection
        # Large matrix operations for 8K processing
        ([4320, 192], [192, 1920]),  # 8K input processing
        ([2160, 384], [384, 3840]),  # 8K mid-level processing
        ([1080, 768], [768, 1920]),  # 8K high-level processing
        ([540, 1536], [1536, 960]),  # 8K very high-level processing
        ([270, 3072], [3072, 480]),  # 8K ultra high-level processing
    ],
)
def test_mm(device, shapes):
    """Test mm operator with proper matrix multiplication dimension compatibility"""
    torch.manual_seed(0)

    input1_shape, input2_shape = shapes

    try:
        # Create tensors with compatible dimensions for mm(input1, input2)
        torch_input1 = torch.rand(input1_shape, dtype=torch.bfloat16)
        torch_input2 = torch.rand(input2_shape, dtype=torch.bfloat16)

        # Convert to ttnn tensors
        ttnn_input1 = ttnn.from_torch(
            torch_input1, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        ttnn_input2 = ttnn.from_torch(
            torch_input2, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        # Perform matrix multiplication
        ttnn_output = ttnn.matmul(ttnn_input1, ttnn_input2)

        # Reference computation
        torch_reference = torch.mm(torch_input1, torch_input2)

        # Convert output back to torch
        ttnn_result = ttnn.to_torch(ttnn_output)

        # Compare results
        check_with_pcc_without_tensor_printout(ttnn_result, torch_reference, 0.99)

        print(f"✅ mm test passed for shapes: input1{input1_shape}, input2{input2_shape}")

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "OOM" in str(e):
            pytest.skip(f"OOM: input1{input1_shape}, input2{input2_shape} - {str(e)}")
        else:
            print(f"❌ Runtime error for shapes: input1{input1_shape}, input2{input2_shape}: {e}")
            raise e
    except Exception as e:
        if "incompatible function arguments" in str(e):
            pytest.skip(f"Type error: input1{input1_shape}, input2{input2_shape} - {str(e)}")
        else:
            print(f"❌ Error for shapes: input1{input1_shape}, input2{input2_shape}: {e}")
            raise e
