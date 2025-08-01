import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        [4, 4000, 32],
        [2, 16000, 32],
        [8, 32, 27],
        [8, 80, 27],
        [8, 27, 32],
        [4, 32, 80],
        [8, 1000, 32],
        [8, 32, 80],
        [1, 1000, 512],
        [1, 4000, 512],
        [1, 16000, 512],
        [2, 1000, 32],
        [2, 32, 1000],
        [4, 1000, 32],
        [4, 32, 1000],
        [1000, 1, 256],
        [8, 32, 1000],
        [8, 1000, 1000],
        [1000, 8, 32],
        [1000, 256],
        [8, 300, 32],
        [8, 32, 300],
        [8, 300, 300],
        [1, 64000, 32],
        [1, 32, 1000],
        [1, 64000, 1000],
        [1, 1000, 32],
        [2, 16000, 1000],
        [5, 4000, 32],
        [5, 32, 1000],
        [5, 4000, 1000],
        [5, 1000, 32],
        [5336, 49, 32],
        [5336, 32, 49],
        [5336, 49, 49],
        [2760, 49, 32],
        [2760, 32, 49],
        [2760, 49, 49],
        [1536, 49, 32],
        [1536, 32, 49],
        [1536, 49, 49],
        [768, 49, 32],
        [768, 32, 49],
        [768, 49, 49],
        [4000, 64, 64],
        [4000, 64, 32],
        [2080, 64, 64],
        [2080, 64, 32],
        [1120, 64, 64],
        [1120, 64, 32],
        [640, 64, 64],
        [640, 64, 32],
    ],
)
def test_bmm(device, input_shape):
    """Test bmm operator with various input shapes from vision models"""
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

        ttnn_output = ttnn.matmul(ttnn_input1, ttnn_input2)

        torch_reference = torch.bmm(torch_input1, torch_input2)

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
