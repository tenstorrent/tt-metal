import pytest
import ttnn
import torch
import torch.nn as nn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_silu_layout_issue(device, use_program_cache, reset_seeds):
    torch_input_tensor = torch.randn(1, 64, 28, 28)
    act = nn.SiLU(inplace=True)
    torch_x = act(torch_input_tensor)
    torch_y1, torch_y2 = torch_x.chunk(2, 1)

    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
        ttnn_input_tensor.shape[3],
    )
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_x = ttnn.silu(ttnn_input_tensor)

    ttnn_x = ttnn.to_layout(ttnn_x, ttnn.ROW_MAJOR_LAYOUT)

    ttnn_y1, ttnn_y2 = ttnn.split(ttnn_x, 2, 3)
    ttnn_y1 = ttnn.to_torch(ttnn_y1)
    ttnn_y1 = ttnn_y1.permute(0, 3, 1, 2)
    ttnn_y1 = ttnn_y1.reshape(torch_y1.shape)

    assert_with_pcc(torch_y1, ttnn_y1, 0.99999)
