import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.cosy_voice.ref.CosyVoice.cosyvoice.transformer.activation import Snake
from models.demos.wormhole.cosy_voice.tt.vocoder.generator import TtSnake


@pytest.mark.parametrize("channels", [512])
@pytest.mark.parametrize("length", [128])
def test_snake_activation(device, channels, length):
    # PyTorch setup
    torch.manual_seed(0)
    pt_snake = Snake(channels, alpha_logscale=False)

    # Input tensor for PyTorch: [B, C, L]
    x_pt = torch.randn(1, channels, length)

    # Run PyTorch
    y_pt_ref = pt_snake(x_pt)

    # TTNN setup
    # TTNN expects tiles, so length should be multiple of 32 for best performance, but let's pad
    # We will reshape [B, C, L] -> [B, 1, L, C] for TTNN
    x_tt_shape = x_pt.transpose(1, 2).unsqueeze(1)  # [1, 1, L, C]

    # Pad to multiple of 32
    padding = (0, 0, 0, (32 - length % 32) % 32)  # (pad_left, pad_right, pad_top, pad_bottom) applied to inner dims? No
    # Since C is 512, L is 128, both are multiples of 32.

    x_tt = ttnn.from_torch(x_tt_shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    alpha_pt = pt_snake.alpha.detach()
    tt_snake = TtSnake(alpha_pt, device)

    # Run TTNN
    y_tt = tt_snake(x_tt)
    y_tt_cpu = ttnn.to_torch(y_tt)

    # Reshape back to [B, C, L]
    y_tt_flat = y_tt_cpu.squeeze(1).transpose(1, 2)

    # Compare
    passing, pcc_msg = comp_pcc(y_pt_ref, y_tt_flat, 0.99)
    print(f"PCC matching: {pcc_msg}")
    assert passing, f"PCC failed: {pcc_msg}"
