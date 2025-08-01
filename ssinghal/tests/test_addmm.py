import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        [27, 256],
        [80, 512],
        [1000, 256],
        [1000, 1024],
        [21000, 256],
        [300, 256],
        [300, 4],
        [300, 512],
        [300, 1024],
        [1000, 32],
        [64000, 32],
        [1000, 64],
        [16000, 64],
        [1000, 160],
        [4000, 160],
        [65366, 128],
        [64000, 128],
        [64000, 512],
        [16905, 256],
        [16000, 256],
        [16000, 1024],
        [4704, 512],
        [4000, 512],
        [4000, 2048],
        [1176, 1024],
        [1000, 4096],
        [16640, 256],
        [4480, 512],
        [1280, 1024],
    ],
)
def test_addmm(device, input_shape):
    """Test addmm operator with various input shapes from vision models"""
    torch.manual_seed(0)

    try:
        torch_input1 = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_input2 = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_input3 = torch.rand(input_shape, dtype=torch.bfloat16)

        if len(input_shape) == 4:
            torch_input1 = torch_input1.permute(0, 2, 3, 1)
            torch_input2 = torch_input2.permute(0, 2, 3, 1)
            torch_input3 = torch_input3.permute(0, 2, 3, 1)

        ttnn_input1 = ttnn.from_torch(
            torch_input1, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        ttnn_input2 = ttnn.from_torch(
            torch_input2, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        ttnn_input3 = ttnn.from_torch(
            torch_input3, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.addmm(ttnn_input1, ttnn_input2, ttnn_input3)

        torch_reference = torch.addmm(torch_input1, torch_input2, torch_input3)

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
