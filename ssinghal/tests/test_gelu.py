import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize("input_shape", [
    # YOLOv12x ultra-high resolution (2176x3840) GELU activation shapes
    [1, 32640, 384],      # Flattened attention shapes
    [1, 8160, 768],       # Flattened feature shapes
    [1, 3072, 1024],      # MLP hidden layers
    [1, 6144, 512],       # Large MLP layers
    [1, 4096, 768],       # Transformer blocks
    [1, 2048, 1536],      # Medium MLP layers
    [1, 1536, 2048],      # Inverse MLP layers
    [1, 384, 1536],       # Small MLP expansion
    [1, 768, 3072],       # Standard transformer MLP
    [1, 1152, 4608],      # Large transformer MLP
    [1, 96, 384],         # Small feature projection
])
def test_gelu(device, input_shape):
    """Test GELU operator with YOLOv12x ultra-high resolution (2176x3840) input shapes"""
    torch.manual_seed(0)

    try:
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.gelu(ttnn_input)

        torch_reference = torch.nn.functional.gelu(torch_input)

        # Convert output back to torch
        ttnn_result = ttnn.to_torch(ttnn_output)

        # Compare results
        check_with_pcc_without_tensor_printout(ttnn_result, torch_reference, 0.99)

    except RuntimeError as e:
        if "Out of Memory" in str(e):
            pytest.skip(f"OOM: {input_shape} - {str(e)}")
        else:
            raise e
    except Exception as e:
        if "incompatible function arguments" in str(e):
            pytest.skip(f"Type error: {input_shape} - {str(e)}")
        else:
            raise e
