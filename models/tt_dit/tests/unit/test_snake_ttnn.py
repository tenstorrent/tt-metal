# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.tt_dit.models.audio_vae.snake_ttnn import Snake


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384}],
    indirect=True,
)
def test_snake_matches_torch(device):
    torch.manual_seed(0)

    batch = 1
    length = 32
    channels = 64

    x = torch.randn(batch, length, channels)

    alpha = torch.ones(1, 1, channels)
    reference = x + torch.sin(x * alpha).pow(2) / (alpha + 1e-9)

    snake = Snake(
        channels=channels,
        mesh_device=device,
        dtype=ttnn.bfloat16,
    )

    state = {}
    snake._prepare_torch_state(state)
    snake.load_torch_state_dict(state)

    x_tt = ttnn.from_torch(
        x,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    output_tt = snake(x_tt)
    output = ttnn.to_torch(output_tt).float()

    error = (output - reference).abs()

    pcc = torch.corrcoef(torch.stack([output.flatten(), reference.flatten()]))[0, 1].item()

    print("PCC:", pcc)
    print("MEAN ERROR:", error.mean().item())
    print("MAX ERROR:", error.max().item())

    assert pcc > 0.999
    assert error.mean().item() < 0.02
