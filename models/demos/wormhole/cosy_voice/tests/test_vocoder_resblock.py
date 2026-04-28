import os
import sys

import pytest
import torch

import ttnn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../ref/CosyVoice")))
from cosyvoice.hifigan.generator import ResBlock

from models.common.utility_functions import comp_pcc
from models.demos.wormhole.cosy_voice.tt.vocoder.generator import TtResBlock


@pytest.mark.parametrize("channels", [256])
@pytest.mark.parametrize("length", [128])
def test_resblock(device, channels, length):
    torch.manual_seed(0)
    batch_size = 1

    # Initialize PyTorch ResBlock
    pt_model = ResBlock(channels=channels, kernel_size=3, dilations=[1, 3, 5], causal=False)
    pt_model.eval()

    # Input [B, C, L]
    x_pt = torch.randn(batch_size, channels, length)

    with torch.no_grad():
        y_pt = pt_model(x_pt)

    # Convert to TTNN ResBlock
    # Manually extract weights and biases to avoid PyTorch parametrizations/weight_norm issues
    state_dict = {}
    for i in range(3):
        state_dict[f"activations1.{i}.alpha"] = pt_model.activations1[i].alpha
        state_dict[f"activations2.{i}.alpha"] = pt_model.activations2[i].alpha
        state_dict[f"convs1.{i}.weight"] = pt_model.convs1[i].weight.data
        state_dict[f"convs1.{i}.bias"] = pt_model.convs1[i].bias.data
        state_dict[f"convs2.{i}.weight"] = pt_model.convs2[i].weight.data
        state_dict[f"convs2.{i}.bias"] = pt_model.convs2[i].bias.data
    # Initialize TTNN ResBlock
    tt_model = TtResBlock(device, state_dict, prefix="")

    # Prepare input for TTNN [B, 1, L, C]
    x_tt_pt = x_pt.transpose(1, 2).unsqueeze(1)
    x_tt = ttnn.from_torch(x_tt_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run TTNN
    y_tt = tt_model(x_tt)

    # Output should be [B, 1, L, C]
    y_tt_cpu = ttnn.to_torch(y_tt)
    # Convert back to [B, C, L]
    y_tt_flat = y_tt_cpu.squeeze(1).transpose(1, 2)

    # Compare
    passing, pcc_msg = comp_pcc(y_pt, y_tt_flat, 0.99)
    print(f"PCC matching: {pcc_msg}")
    assert passing, f"PCC failed: {pcc_msg}"
