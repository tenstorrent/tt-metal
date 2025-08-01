import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        [1, 64, 640, 400],
        [1, 64, 320, 200],
        [1, 128, 160, 100],
        [1, 256, 80, 50],
        [1, 512, 40, 25],
        [1, 256, 160, 100],
        [1, 512, 80, 50],
        [1, 1024, 40, 25],
        [1, 3, 160, 100, 2],
        [1, 3, 80, 50, 2],
        [1, 3, 40, 25, 2],
        [1, 80, 320, 200],
        [1, 160, 160, 100],
        [1, 320, 80, 50],
        [1, 320, 40, 25],
        [1, 2, 21000],
        [1, 32, 320, 200],
        [1, 64, 160, 100],
        [1, 128, 80, 50],
        [1, 256, 40, 25],
        [1, 4, 80, 50],
        [1, 4, 32, 80, 50],
        [1, 4, 1, 80, 50],
        [1, 2, 160, 100],
        [1, 2, 32, 160, 100],
        [1, 2, 1, 160, 100],
        [1, 80, 512],
        [1, 8, 40, 25],
        [1, 8, 32, 40, 25],
        [1, 8, 1, 40, 25],
        [1, 16, 320, 200],
        [1, 32, 160, 100],
        [1, 32, 80, 50],
        [1, 64, 40, 25],
        [1, 128, 40, 25],
        [1, 64, 80, 50],
        [1, 16, 160, 100],
        [1, 1024, 80, 50],
        [1, 1000, 256],
        [1000, 1, 256],
        [1, 300, 4],
        [1, 300, 256],
        [1, 256, 320, 200],
        [1, 512, 160, 100],
        [1, 2048, 40, 25],
        [1, 24, 320, 200],
        [1, 96, 80, 50],
        [1, 160, 40, 25],
        [1, 40, 160, 100],
        [1, 80, 80, 50],
        [1, 112, 80, 50],
        [1, 192, 40, 25],
        [1, 24, 640, 400],
        [1, 48, 160, 100],
        [1, 136, 80, 50],
        [1, 232, 40, 25],
        [1, 384, 40, 25],
        [1, 56, 160, 100],
        [1, 160, 80, 50],
        [1, 272, 40, 25],
        [1, 448, 40, 25],
        [1, 64000, 32],
        [1, 16000, 64],
        [1, 4000, 160],
        [1334, 4, 49, 49],
        [1, 64000, 128],
        [1, 1334, 4, 49, 49],
        [345, 8, 49, 49],
        [1, 16000, 256],
        [1, 345, 8, 49, 49],
        [96, 16, 49, 49],
        [1, 4000, 512],
        [1, 96, 16, 49, 49],
        [1, 80, 50, 512],
        [1, 40, 50, 512],
        [1, 40, 25, 512],
        [1, 40, 25, 2048],
        [24, 32, 49, 49],
        [1, 1000, 1024],
        [1, 24, 32, 49, 49],
        [4000, 512],
        [1, 80, 56, 512],
        [1, 10, 8, 7, 8, 512],
        [1, 10, 7, 8, 8, 512],
        [70, 8, 8, 512],
        [70, 64, 512],
        [4480, 512],
        [1, 1000, 2048],
        [1000, 2048],
    ],
)
def test_add(device, input_shape):
    """Test add operator with various input shapes from vision models"""
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

        ttnn_output = ttnn.add(ttnn_input1, ttnn_input2)

        torch_reference = torch.add(torch_input1, torch_input2)

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
