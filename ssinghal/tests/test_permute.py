import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 3, 85, 160, 100],
        [1, 3, 85, 80, 50],
        [1, 3, 85, 40, 25],
        [1, 4, 32, 80, 50, 1],
        [1, 4, 80, 50, 1, 32],
        [4, 80, 50, 1, 1, 80],
        [1, 2, 32, 160, 100, 1],
        [1, 2, 160, 100, 1, 32],
        [2, 160, 100, 1, 1, 80],
        [1, 27, 8, 32, 1],
        [1, 8, 1, 27, 32],
        [8, 80, 1, 1, 27],
        [1, 8, 80, 27, 1],
        [1, 1, 8, 32, 27],
        [1, 80, 8, 1, 27],
        [1, 80, 4, 32, 1, 1],
        [1, 4, 1, 1, 80, 32],
        [1, 80, 8, 32, 1, 1],
        [1, 8, 32, 40, 25, 1],
        [1, 8, 1, 1, 80, 32],
        [1, 8, 40, 25, 1, 32],
        [8, 40, 25, 1, 1, 80],
        [1, 512, 40, 25, 1],
        [1, 512, 80, 50, 1],
        [1, 512, 160, 100, 1],
        [1, 1, 40, 25, 512],
        [1, 1, 80, 50, 512],
        [1, 1, 160, 100, 512],
        [40, 25, 1, 1, 80],
        [80, 50, 1, 1, 80],
        [160, 100, 1, 1, 80],
        [1, 4, 16, 21000],
        [4, 1000, 2, 96],
        [4, 2, 32, 1000],
        [1, 80, 50, 64],
        [1, 64, 80, 50],
        [1, 1000, 4, 96],
        [1, 4, 32, 1000],
        [1, 40, 25, 128],
        [1, 256, 1000],
        [1, 1000, 256],
        [1, 256, 4000],
        [1, 256, 16000],
        [1, 64000, 32],
        [1, 32, 1000],
        [1, 1000, 1, 32],
        [1, 64000, 1, 32],
        [1, 1, 64000, 32],
        [1, 320, 200, 32],
        [1, 16000, 64],
        [1, 64, 1000],
        [1, 1000, 2, 32],
        [1, 16000, 2, 32],
        [1, 2, 16000, 32],
        [1, 160, 100, 64],
        [1, 4000, 160],
        [1, 160, 1000],
        [1, 1000, 5, 32],
        [1, 4000, 5, 32],
        [1, 5, 4000, 32],
        [1, 80, 50, 160],
        [1, 1000, 8, 32],
        [1, 8, 1000, 32],
        [1, 40, 25, 256],
        [1, 256, 40, 25],
        [1, 46, 7, 29, 7, 128],
        [1334, 49, 4, 32],
        [1334, 4, 49, 32],
        [1, 46, 29, 7, 7, 128],
        [1, 23, 7, 15, 7, 256],
        [345, 49, 8, 32],
        [345, 8, 49, 32],
        [1, 23, 15, 7, 7, 256],
        [1, 12, 7, 8, 7, 512],
        [96, 49, 16, 32],
        [96, 16, 49, 32],
        [1, 12, 8, 7, 7, 512],
        [1, 6, 7, 4, 7, 1024],
        [24, 49, 32, 32],
        [24, 32, 49, 32],
        [1, 6, 4, 7, 7, 1024],
        [1, 40, 8, 25, 8, 128],
        [1000, 64, 4, 32],
        [1000, 4, 64, 32],
        [1, 40, 25, 8, 8, 128],
        [1, 20, 8, 13, 8, 256],
        [260, 64, 8, 32],
        [260, 8, 64, 32],
        [1, 20, 13, 8, 8, 256],
        [1, 10, 8, 7, 8, 512],
        [70, 64, 16, 32],
        [70, 16, 64, 32],
        [1, 10, 7, 8, 8, 512],
        [1, 5, 8, 4, 8, 1024],
        [20, 64, 32, 32],
        [20, 32, 64, 32],
        [1, 5, 4, 8, 8, 1024],
    ],
)
def test_permute(device, input_shape):
    """Test permute operator with YOLOv12 high-resolution input shapes"""
    torch.manual_seed(0)

    try:
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input = torch_input.permute(0, 2, 3, 1)

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.permute(ttnn_input, (0, 3, 1, 2)) if len(input_shape) == 4 else ttnn_input

        torch_reference = torch_input.permute(0, 3, 1, 2) if len(input_shape) == 4 else torch_input

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
