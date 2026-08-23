# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.tt_dit.models.audio_vae.hift_resblock_ttnn import HiFTResBlock


def snake_torch(x, alpha):
    return x + torch.sin(x * alpha).pow(2) / (alpha + 1e-9)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 64 * 1024}],
    indirect=True,
)
def test_hift_resblock_matches_torch(device):
    torch.manual_seed(0)

    batch = 1
    length = 32
    channels = 32
    kernel_size = 3
    dilations = (1, 3, 5)

    x = torch.randn(batch, length, channels) * 0.5

    block = HiFTResBlock(
        channels,
        kernel_size=kernel_size,
        dilations=dilations,
        causal=False,
        mesh_device=device,
        dtype=ttnn.bfloat16,
    )

    weights = []

    for i, dilation in enumerate(dilations):
        w1 = torch.randn(channels, channels, kernel_size) * 0.03
        b1 = torch.randn(channels) * 0.01

        w2 = torch.randn(channels, channels, kernel_size) * 0.03
        b2 = torch.randn(channels) * 0.01

        weights.append((w1, b1, w2, b2))

        block.convs1[i].load_torch_state_dict({"weight": w1.clone(), "bias": b1.clone()})

        block.convs2[i].load_torch_state_dict({"weight": w2.clone(), "bias": b2.clone()})

        block.activations1[i].load_torch_state_dict({})
        block.activations2[i].load_torch_state_dict({})

    # PyTorch reference uses BCT.
    reference = x.transpose(1, 2)
    alpha = torch.ones(1, channels, 1)

    for i, dilation in enumerate(dilations):
        w1, b1, w2, b2 = weights[i]

        xt = snake_torch(reference, alpha)
        xt = F.conv1d(
            xt,
            w1,
            b1,
            stride=1,
            padding=dilation,
            dilation=dilation,
        )

        xt = snake_torch(xt, alpha)
        xt = F.conv1d(
            xt,
            w2,
            b2,
            stride=1,
            padding=1,
            dilation=1,
        )

        reference = reference + xt

    reference = reference.transpose(1, 2)

    x_tt = ttnn.from_torch(
        x,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
    )

    output_tt = block(x_tt)
    output = ttnn.to_torch(output_tt).float()

    error = (output - reference).abs()

    pcc = torch.corrcoef(torch.stack([output.flatten(), reference.flatten()]))[0, 1].item()

    print("PCC:", pcc)
    print("MEAN ERROR:", error.mean().item())
    print("MAX ERROR:", error.max().item())

    assert pcc > 0.995
    assert error.mean().item() < 0.03
