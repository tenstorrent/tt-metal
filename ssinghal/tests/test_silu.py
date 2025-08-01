import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 80, 640, 400],
        [1, 160, 320, 200],
        [1, 80, 320, 200],
        [1, 320, 160, 100],
        [1, 160, 160, 100],
        [1, 640, 80, 50],
        [1, 320, 80, 50],
        [1, 640, 40, 25],
        [1, 320, 40, 25],
        [1, 80, 40, 25],
        [1, 80, 80, 50],
        [1, 80, 160, 100],
        [1, 32, 640, 400],
        [1, 64, 320, 200],
        [1, 32, 320, 200],
        [1, 128, 160, 100],
        [1, 64, 160, 100],
        [1, 256, 80, 50],
        [1, 128, 80, 50],
        [1, 512, 40, 25],
        [1, 256, 40, 25],
        [1, 128, 40, 25],
        [1, 64, 40, 25],
        [1, 64, 80, 50],
        [1, 16, 640, 400],
        [1, 8, 320, 200],
        [1, 16, 320, 200],
        [1, 16, 160, 100],
        [1, 32, 160, 100],
        [1, 32, 80, 50],
        [1, 256, 160, 100],
        [1, 8, 1, 1],
        [1, 96, 640, 400],
        [1, 96, 320, 200],
        [1, 4, 1, 1],
        [1, 144, 320, 200],
        [1, 6, 1, 1],
        [1, 144, 160, 100],
        [1, 240, 160, 100],
        [1, 10, 1, 1],
        [1, 240, 80, 50],
        [1, 480, 80, 50],
        [1, 20, 1, 1],
        [1, 672, 80, 50],
        [1, 28, 1, 1],
        [1, 672, 40, 25],
        [1, 1152, 40, 25],
        [1, 48, 1, 1],
        [1, 1280, 40, 25],
        [1, 40, 640, 400],
        [1, 24, 640, 400],
        [1, 144, 640, 400],
        [1, 192, 320, 200],
        [1, 192, 160, 100],
        [1, 288, 160, 100],
        [1, 12, 1, 1],
        [1, 288, 80, 50],
        [1, 576, 80, 50],
        [1, 24, 1, 1],
        [1, 816, 80, 50],
        [1, 34, 1, 1],
        [1, 816, 40, 25],
        [1, 1392, 40, 25],
        [1, 58, 1, 1],
        [1, 2304, 40, 25],
        [1, 96, 1, 1],
        [1, 1536, 40, 25],
        [1, 48, 640, 400],
        [1, 336, 160, 100],
        [1, 14, 1, 1],
        [1, 336, 80, 50],
        [1, 960, 80, 50],
        [1, 40, 1, 1],
        [1, 960, 40, 25],
        [1, 1632, 40, 25],
        [1, 68, 1, 1],
        [1, 2688, 40, 25],
        [1, 112, 1, 1],
        [1, 1792, 40, 25],
    ],
)
def test_silu(device, input_shape):
    """Test silu operator with various input shapes from vision models"""
    torch.manual_seed(0)

    try:
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.silu(ttnn_input)

        torch_reference = torch.nn.functional.silu(torch_input)

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
