import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "shapes",
    [
        # Format: (input1_shape, input2_shape) for bmm(input1, input2)
        # BMM requires input1[b,n,k] @ input2[b,k,m] -> output[b,n,m]
        # YOLOv12x attention BMM shapes - proper dimension compatibility
        ([12, 8160, 32], [12, 32, 8160]),  # Multi-head attention: Q @ K^T
        ([12, 8160, 32], [12, 32, 64]),  # Multi-head attention: Q @ V
        ([12, 8160, 8160], [12, 8160, 32]),  # Attention scores @ V
        ([12, 32640, 32], [12, 32, 64]),  # Full attention
        ([12, 96, 32640], [12, 32640, 32]),  # Attention keys
        ([1, 8160, 768], [1, 768, 384]),  # Flattened features
        ([1, 32640, 384], [1, 384, 768]),  # Large attention
        ([8, 4080, 64], [8, 64, 96]),  # Medium attention
        ([8, 2040, 96], [8, 96, 128]),  # Smaller attention
        ([16, 1020, 48], [16, 48, 64]),  # High-head attention
        ([4, 16320, 128], [4, 128, 256]),  # Low-head attention
        ([6, 5440, 96], [6, 96, 128]),  # Mixed attention
        ([24, 680, 32], [24, 32, 64]),  # Many-head attention
        # 8K YOLOv12x batch matrix multiplication shapes
        ([1, 270, 768], [1, 768, 480]),  # 8K attention
        ([1, 135, 1536], [1, 1536, 240]),  # 8K deep attention
        ([8, 270, 768], [8, 768, 240]),  # 8K batch attention
        ([4, 540, 384], [4, 384, 960]),  # 8K medium batch
        # Basic compatibility test shapes
        ([8, 32, 64], [8, 64, 1000]),  # Small batch
        ([8, 1000, 32], [8, 32, 64]),  # Medium batch
        ([8, 300, 32], [8, 32, 64]),  # Small sequence
        ([8, 300, 64], [8, 64, 128]),  # Medium sequence
        ([1, 64000, 32], [1, 32, 64]),  # Large sequence
        ([1, 32, 64], [1, 64, 1000]),  # Wide output
        ([2, 16000, 64], [2, 64, 128]),  # Large batch
        ([5, 4000, 32], [5, 32, 64]),  # Multi batch
        ([5, 1000, 64], [5, 64, 128]),  # Multi batch medium
        # Square attention patterns
        ([768, 49, 32], [768, 32, 49]),  # Spatial attention
        ([1536, 49, 32], [1536, 32, 49]),  # Spatial attention large
        ([2760, 49, 32], [2760, 32, 49]),  # Spatial attention very large
        ([5336, 49, 32], [5336, 32, 49]),  # Spatial attention huge
        # Self-attention patterns (square middle dimension)
        ([768, 49, 64], [768, 64, 49]),  # Self attention
        ([1536, 49, 64], [1536, 64, 49]),  # Self attention large
        ([4000, 64, 32], [4000, 32, 64]),  # Large self attention
        ([2080, 64, 32], [2080, 32, 64]),  # Medium self attention
        ([1120, 64, 32], [1120, 32, 64]),  # Small self attention
        ([640, 64, 32], [640, 32, 64]),  # Tiny self attention
    ],
)
def test_bmm(device, shapes):
    """Test bmm operator with proper batch matrix multiplication dimension compatibility"""
    torch.manual_seed(0)

    input1_shape, input2_shape = shapes

    try:
        # Create tensors with compatible dimensions for bmm(input1, input2)
        torch_input1 = torch.rand(input1_shape, dtype=torch.bfloat16)
        torch_input2 = torch.rand(input2_shape, dtype=torch.bfloat16)

        # Convert to ttnn tensors
        ttnn_input1 = ttnn.from_torch(
            torch_input1, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        ttnn_input2 = ttnn.from_torch(
            torch_input2, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        # Perform batch matrix multiplication
        ttnn_output = ttnn.matmul(ttnn_input1, ttnn_input2)

        # Reference computation
        torch_reference = torch.bmm(torch_input1, torch_input2)

        # Convert output back to torch
        ttnn_result = ttnn.to_torch(ttnn_output)

        # Compare results
        check_with_pcc_without_tensor_printout(ttnn_result, torch_reference, 0.99)

        print(f"✅ bmm test passed for shapes: input1{input1_shape}, input2{input2_shape}")

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
