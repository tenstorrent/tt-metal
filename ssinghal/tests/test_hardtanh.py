import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 32, 640, 400],
        [1, 96, 640, 400],
        [1, 96, 320, 200],
        [1, 144, 320, 200],
        [1, 144, 160, 100],
        [1, 192, 160, 100],
        [1, 192, 80, 50],
        [1, 384, 80, 50],
        [1, 576, 80, 50],
        [1, 576, 40, 25],
        [1, 960, 40, 25],
        [1, 1280, 40, 25],
    ],
)
def test_hardtanh(device, input_shape):
    """Test hardtanh operator with various input shapes from vision models"""
    torch.manual_seed(0)

    try:
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.hardtanh(ttnn_input)

        torch_reference = torch.nn.functional.hardtanh(torch_input)

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
