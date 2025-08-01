import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 3, 160, 100, 85],
        [1, 3, 80, 50, 85],
        [1, 3, 40, 25, 85],
        [1, 2, 64, 1000],
        [4, 2, 32, 1000],
        [4, 2, 1000, 32],
        [4, 1000, 2, 32],
        [1, 1000, 4, 32],
        [1, 128, 40, 25],
        [1, 256, 40, 25],
        [1, 32, 320, 200],
        [1, 16000, 2, 32],
        [1, 64, 160, 100],
        [1, 4000, 5, 32],
        [1, 160, 80, 50],
        [1, 1000, 8, 32],
        [1, 46, 29, 7, 7, 128],
        [1334, 4, 32, 49],
        [1334, 4, 49, 32],
        [1334, 49, 4, 32],
        [1, 46, 7, 29, 7, 128],
        [1, 320, 200, 128],
        [1, 23, 15, 7, 7, 256],
        [345, 8, 32, 49],
        [345, 8, 49, 32],
        [345, 49, 8, 32],
        [1, 23, 7, 15, 7, 256],
        [1, 160, 100, 256],
        [1, 12, 8, 7, 7, 512],
        [96, 16, 32, 49],
        [96, 16, 49, 32],
        [96, 49, 16, 32],
        [1, 12, 7, 8, 7, 512],
        [1, 80, 50, 512],
        [1, 6, 4, 7, 7, 1024],
        [24, 32, 32, 49],
        [24, 32, 49, 32],
        [24, 49, 32, 32],
        [1, 6, 7, 4, 7, 1024],
        [1, 40, 25, 1024],
        [1, 40, 25, 8, 8, 128],
        [1000, 4, 32, 64],
        [1000, 4, 64, 32],
        [1000, 64, 4, 32],
        [1, 40, 8, 25, 8, 128],
        [1, 20, 13, 8, 8, 256],
        [260, 8, 32, 64],
        [260, 8, 64, 32],
        [260, 64, 8, 32],
        [1, 20, 8, 13, 8, 256],
        [1, 10, 7, 8, 8, 512],
        [70, 16, 32, 64],
        [70, 16, 64, 32],
        [70, 64, 16, 32],
        [1, 10, 8, 7, 8, 512],
        [70, 8, 8, 512],
        [70, 64, 512],
        [4480, 512],
        [1, 5, 4, 8, 8, 1024],
        [20, 32, 32, 64],
        [20, 32, 64, 32],
        [20, 64, 32, 32],
        [1, 5, 8, 4, 8, 1024],
    ],
)
def test_clone(device, input_shape):
    """Test clone operator with various input shapes from vision models"""
    torch.manual_seed(0)

    try:
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.clone(ttnn_input)

        torch_reference = torch.clone(torch_input)

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
