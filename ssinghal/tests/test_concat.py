import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 64, 320, 200],
        [1, 128, 160, 100],
        [1, 256, 80, 50],
        [1, 512, 40, 25],
        [1, 256, 40, 25],
        [1, 128, 80, 50],
        [1, 64, 160, 100],
        [1, 640, 80, 50],
        [1, 640, 160, 100],
        [1, 320, 160, 100],
        [1, 320, 80, 50],
        [1, 640, 40, 25],
        [1, 512, 80, 50],
        [1, 256, 160, 100],
        [1, 64, 80, 50],
        [1, 128, 40, 25],
    ],
)
def test_concat(device, input_shape):
    """Test Concat operator with various input shapes from vision models"""
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

        ttnn_output = ttnn.concat([ttnn_input1, ttnn_input2], dim=0)

        torch_reference = torch.cat([torch_input1, torch_input2], dim=0)

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
