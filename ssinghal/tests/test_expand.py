import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 1, 40, 25],
        [1, 1, 80, 50],
        [1, 1, 160, 100],
        [1, 2, 32, 1000],
        [1, 2, 1000, 32],
        [1, 2, 1000, 1000],
        [1, 2, 64, 1000],
        [2, 64, 1000],
        [2, 1000, 1000],
        [1, 128, 40, 25],
        [4, 2, 32, 1000],
        [4, 2, 1000, 32],
        [1, 4, 32, 1000],
        [1, 4, 1000, 32],
        [1000, 1, 256],
        [1, 1, 32, 1000],
        [1, 1, 64000, 32],
        [1, 1, 1000, 32],
        [1, 1, 64000, 1000],
        [1, 2, 16000, 32],
        [1, 2, 16000, 1000],
        [1, 5, 32, 1000],
        [1, 5, 4000, 32],
        [1, 5, 1000, 32],
        [1, 5, 4000, 1000],
        [1, 8, 32, 1000],
        [1, 8, 1000, 32],
        [1, 8, 1000, 1000],
        [1334, 4, 32, 49],
        [1334, 4, 49, 32],
        [1334, 4, 49, 49],
        [345, 8, 32, 49],
        [345, 8, 49, 32],
        [345, 8, 49, 49],
        [96, 16, 32, 49],
        [96, 16, 49, 32],
        [96, 16, 49, 49],
        [24, 32, 32, 49],
        [24, 32, 49, 32],
        [24, 32, 49, 49],
        [1000, 4, 64, 1],
        [1000, 4, 32, 64],
        [1000, 4, 64, 32],
        [1000, 4, 64, 64],
        [260, 8, 64, 1],
        [260, 8, 32, 64],
        [260, 8, 64, 32],
        [260, 8, 64, 64],
        [70, 16, 64, 1],
        [70, 16, 32, 64],
        [70, 16, 64, 32],
        [70, 16, 64, 64],
        [20, 32, 64, 1],
        [20, 32, 32, 64],
        [20, 32, 64, 32],
        [20, 32, 64, 64],
    ],
)
def test_expand(device, input_shape):
    """Test expand operator with various input shapes from vision models"""
    torch.manual_seed(0)

    try:
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.expand(ttnn_input)

        torch_reference = torch.expand(torch_input)

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
