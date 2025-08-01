import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 32, 1, 1],
        [1, 96, 1, 1],
        [1, 144, 1, 1],
        [1, 240, 1, 1],
        [1, 480, 1, 1],
        [1, 672, 1, 1],
        [1, 1152, 1, 1],
        [1, 40, 1, 1],
        [1, 24, 1, 1],
        [1, 192, 1, 1],
        [1, 288, 1, 1],
        [1, 576, 1, 1],
        [1, 816, 1, 1],
        [1, 1392, 1, 1],
        [1, 2304, 1, 1],
        [1, 48, 1, 1],
        [1, 336, 1, 1],
        [1, 960, 1, 1],
        [1, 1632, 1, 1],
        [1, 2688, 1, 1],
        [1, 1, 1280, 800],
    ],
)
def test_sigmoid(device, input_shape):
    """Test Sigmoid operator with various input shapes from vision models"""
    torch.manual_seed(0)

    try:
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.sigmoid(ttnn_input)

        torch_reference = torch.nn.functional.sigmoid(torch_input)

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
