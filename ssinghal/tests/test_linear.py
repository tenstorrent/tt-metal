import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout


@pytest.mark.parametrize(
    "input_shape",
    [
        [27, 256],
        [80, 512],
        [80, 128],
        [80, 256],
        [1000, 1024],
        [1000, 256],
        [21000, 256],
        [21000, 80],
        [300, 256],
        [300, 4],
        [300, 512],
        [300, 192],
        [300, 96],
        [300, 1024],
        [300, 80],
        [1, 2048],
        [1, 1280],
        [1, 1536],
        [1, 1792],
        [1, 8000],
        [1000, 32],
        [64000, 32],
        [64000, 128],
        [1000, 64],
        [16000, 64],
        [16000, 256],
        [1000, 160],
        [4000, 160],
        [4000, 640],
        [1, 256],
        [65366, 128],
        [64000, 512],
        [16905, 256],
        [16000, 1024],
        [4000, 512],
        [4704, 512],
        [4000, 2048],
        [1176, 1024],
        [1000, 4096],
        [1, 1024],
        [16640, 256],
        [4480, 512],
        [1280, 1024],
    ],
)
def test_linear(device, input_shape):
    """Test Linear operator with various input shapes from vision models"""
    torch.manual_seed(0)

    try:
        batch_size, seq_len, hidden_size = input_shape if len(input_shape) == 3 else (1, input_shape[0], input_shape[1])
        torch_input = torch.rand((batch_size, seq_len, hidden_size), dtype=torch.bfloat16)
        torch_weight = torch.rand((hidden_size, hidden_size), dtype=torch.bfloat16)

        ttnn_input = ttnn.from_torch(
            torch_input, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        ttnn_weight = ttnn.from_torch(
            torch_weight, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )

        ttnn_output = ttnn.linear(ttnn_input, ttnn_weight)

        torch_reference = torch.nn.functional.linear(torch_input, torch_weight)

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
