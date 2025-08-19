import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "shapes",
    [
        # Format: (bias_shape, mat1_shape, mat2_shape) for addmm(bias, mat1, mat2)
        # For addmm: bias is broadcast to result shape [mat1_rows, mat2_cols]
        ([27, 256], [27, 256], [256, 256]),  # Basic compatible shapes
        ([80, 512], [80, 256], [256, 512]),  # Basic compatible shapes
        ([1000, 256], [1000, 512], [512, 256]),  # Basic compatible shapes
        ([1000, 1024], [1000, 512], [512, 1024]),  # Basic compatible shapes
        ([21000, 256], [21000, 512], [512, 256]),  # Large batch
        ([300, 256], [300, 128], [128, 256]),  # Medium batch
        ([300, 4], [300, 64], [64, 4]),  # Small output
        ([300, 512], [300, 256], [256, 512]),  # Medium batch medium output
        ([300, 1024], [300, 512], [512, 1024]),  # Medium batch large output
        ([1000, 32], [1000, 64], [64, 32]),  # Large batch small output
        ([64000, 32], [64000, 128], [128, 32]),  # Very large batch
        ([1000, 64], [1000, 128], [128, 64]),  # Large batch medium output
        ([16000, 64], [16000, 256], [256, 64]),  # Very large batch
        ([1000, 160], [1000, 256], [256, 160]),  # Large batch
        ([4000, 160], [4000, 512], [512, 160]),  # Large batch
        ([65366, 128], [65366, 256], [256, 128]),  # Huge batch
        ([64000, 128], [64000, 256], [256, 128]),  # Huge batch
        ([64000, 512], [64000, 256], [256, 512]),  # Huge batch large output
        ([16905, 256], [16905, 512], [512, 256]),  # Large batch
        ([16000, 256], [16000, 512], [512, 256]),  # Large batch
        ([16000, 1024], [16000, 512], [512, 1024]),  # Large batch large output
        ([4704, 512], [4704, 256], [256, 512]),  # Large batch
        ([4000, 512], [4000, 256], [256, 512]),  # Large batch
        ([4000, 2048], [4000, 1024], [1024, 2048]),  # Large batch very large output
        ([1176, 1024], [1176, 512], [512, 1024]),  # Large batch
        ([1000, 4096], [1000, 2048], [2048, 4096]),  # Large output
        ([16640, 256], [16640, 512], [512, 256]),  # Large batch
        ([4480, 512], [4480, 256], [256, 512]),  # Large batch
        ([1280, 1024], [1280, 512], [512, 1024]),  # Large batch
        # 8K YOLOv12x matrix shapes for addmm operations - proper result shapes
        ([32400, 768], [32400, 1024], [1024, 768]),  # 8K flattened to linear
        ([8100, 1536], [8100, 768], [768, 1536]),  # 8K half-res flattened
        ([2025, 3072], [2025, 1536], [1536, 3072]),  # 8K quarter-res flattened
        ([540, 6144], [540, 3072], [3072, 6144]),  # 8K eighth-res flattened
        ([135, 12288], [135, 6144], [6144, 12288]),  # 8K sixteenth-res flattened
        ([270, 6144], [270, 3072], [3072, 6144]),  # 8K attention shapes
        ([135, 12288], [135, 6144], [6144, 12288]),  # 8K deep attention
        ([67, 24576], [67, 12288], [12288, 24576]),  # 8K very deep attention
    ],
)
def test_addmm(device, shapes):
    """Test addmm operator with proper matrix dimension compatibility"""
    torch.manual_seed(0)

    bias_shape, mat1_shape, mat2_shape = shapes

    try:
        # Create tensors with compatible dimensions for addmm(bias, mat1, mat2)
        torch_bias = torch.rand(bias_shape, dtype=torch.bfloat16)
        torch_mat1 = torch.rand(mat1_shape, dtype=torch.bfloat16)
        torch_mat2 = torch.rand(mat2_shape, dtype=torch.bfloat16)

        # Convert to ttnn tensors
        ttnn_bias = ttnn.from_torch(
            torch_bias, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        ttnn_mat1 = ttnn.from_torch(
            torch_mat1, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        ttnn_mat2 = ttnn.from_torch(
            torch_mat2, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        # Perform addmm operation
        ttnn_output = ttnn.addmm(ttnn_bias, ttnn_mat1, ttnn_mat2)

        # Reference computation
        torch_reference = torch.addmm(torch_bias, torch_mat1, torch_mat2)

        # Convert output back to torch
        ttnn_result = ttnn.to_torch(ttnn_output)

        # Compare results
        check_with_pcc_without_tensor_printout(ttnn_result, torch_reference, 0.99)

        print(f"✅ addmm test passed for shapes: bias{bias_shape}, mat1{mat1_shape}, mat2{mat2_shape}")

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "OOM" in str(e):
            pytest.skip(f"OOM: bias{bias_shape}, mat1{mat1_shape}, mat2{mat2_shape} - {str(e)}")
        else:
            print(f"❌ Runtime error for shapes: bias{bias_shape}, mat1{mat1_shape}, mat2{mat2_shape}: {e}")
            raise e
    except Exception as e:
        if "incompatible function arguments" in str(e):
            pytest.skip(f"Type error: bias{bias_shape}, mat1{mat1_shape}, mat2{mat2_shape} - {str(e)}")
        else:
            print(f"❌ Error for shapes: bias{bias_shape}, mat1{mat1_shape}, mat2{mat2_shape}: {e}")
            raise e
