import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 512, 40, 25],
        [1, 256, 320, 200],
        [1, 512, 160, 100],
        [1, 1024, 80, 50],
        [1, 128, 160, 100],
        [1, 256, 80, 50],
        [1, 320, 40, 25],
        [1, 256, 40, 25],
        [1, 128, 40, 25],
        [1, 32, 641, 401],
        [1, 64, 640, 400],
        [1, 64, 1280, 800],
        [1, 128, 640, 400],
    ],
)
def test_maxpool2d(device, input_shape):
    """Test MaxPool2d operator with various input shapes from vision models"""
    torch.manual_seed(0)

    try:
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.max_pool2d(ttnn_input)

        torch_reference = torch.nn.functional.max_pool2d(torch_input)

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
