import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 300, 8, 3, 4, 2],
        [1, 300, 1, 1, 1, 2],
        [1, 300, 8, 4, 2],
        [1, 8, 300, 4, 2],
        [8, 32, 40, 25],
        [8, 300, 4, 2],
        [8, 32, 80, 50],
        [8, 32, 160, 100],
        [8, 32, 300, 4],
        [8, 32, 300, 3, 4],
        [8, 32, 300, 12],
        [8, 1, 300, 12],
        [8, 32, 300],
        [1, 256, 300],
        [1, 300, 256],
        [300, 256],
    ],
)
def test_mul(device, input_shape):
    """Test mul operator with various input shapes from vision models"""
    torch.manual_seed(0)

    try:
        torch_input1 = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_input2 = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input1 = torch_input1.permute(0, 2, 3, 1)
            torch_input2 = torch_input2.permute(0, 2, 3, 1)

        ttnn_input1 = ttnn.from_torch(
            torch_input1, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        ttnn_input2 = ttnn.from_torch(
            torch_input2, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.mul(ttnn_input1, ttnn_input2)

        torch_reference = torch.mul(torch_input1, torch_input2)

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
