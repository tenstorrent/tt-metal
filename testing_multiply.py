import torch
import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test(device, reset_seeds):
    input_2d = torch.tensor([[1000.0], [1000.0]], dtype=torch.bfloat16)
    input_128 = torch.randn(1, 128, dtype=torch.bfloat16)
    torch_output = input_2d * input_128

    ttnn_input_2d = ttnn.from_torch(input_2d, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    ttnn_input_128 = ttnn.from_torch(input_128, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    ttnn_output = ttnn.multiply(ttnn_input_2d, ttnn_input_128)

    assert_with_pcc(torch_output, ttnn.to_torch(ttnn_output), 1)  # 0.7068334416472716
