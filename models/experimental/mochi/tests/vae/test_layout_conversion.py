import math

import pytest
import torch
import ttnn

from models.experimental.mochi.common import compute_metrics


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 12, 28, 60, 106), 768, (1, 1, 1), (1, 1, 1), (0, 0, 0), "zeros"],
        [(1, 768, 28, 60, 106), 768, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 512, 82, 120, 212), 512, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 256, 163, 240, 424), 256, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 128, 163, 480, 848), 128, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
    ],
    ids=["variant0", "variant1", "variant2", "variant3", "variant4"],
)
@pytest.mark.parametrize("parallel_factor", [8])
def test_layout_conversion(
    device, input_shape, out_channels, kernel_size, stride, padding, padding_mode, parallel_factor
):
    B, C, D, H, W = input_shape
    D = math.ceil(D / parallel_factor)
    input_shape = (B, C, D, H, W)
    input_tensor = torch.randn(*input_shape, dtype=torch.float32)
    input_tensor = input_tensor.permute(0, 2, 3, 4, 1).reshape(B, D, H * W, C)
    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_input = ttnn.to_layout(tt_input, ttnn.ROW_MAJOR_LAYOUT)
    tt_input = ttnn.to_layout(tt_input, ttnn.TILE_LAYOUT)
    tt_input = ttnn.to_layout(tt_input, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.to_torch(tt_input).float()
    pcc, mse, mae = compute_metrics(tt_output, input_tensor)
    assert pcc > 0.99
