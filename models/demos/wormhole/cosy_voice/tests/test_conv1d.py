import pytest
import torch
import torch.nn as nn

import ttnn
from models.common.utility_functions import comp_pcc


@pytest.mark.parametrize("in_channels", [256])
@pytest.mark.parametrize("out_channels", [256])
@pytest.mark.parametrize("kernel_size", [3])
@pytest.mark.parametrize("dilation", [1, 3])
@pytest.mark.parametrize("length", [128])
def test_conv1d(device, in_channels, out_channels, kernel_size, dilation, length):
    torch.manual_seed(0)
    batch_size = 1
    # Padding formula used in CosyVoice
    padding = (kernel_size - 1) * dilation // 2

    # PyTorch Setup
    pt_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation)

    # Input for PyTorch: [B, C, L]
    x_pt = torch.randn(batch_size, in_channels, length)
    y_pt = pt_conv(x_pt)

    # TTNN Setup
    # TTNN expects input in [N, H, W, C] -> [Batch, 1, Length, Channels]
    # We'll put it on device, TTNN's conv expects ROW_MAJOR or TILE layout?
    # Usually ROW_MAJOR or TILE for activation, the wrapper prepares it if it's on host or device.
    x_tt = x_pt.transpose(1, 2).unsqueeze(1)  # [B, 1, L, C]
    x_tt = ttnn.from_torch(x_tt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Weights: [out_channels, in_channels, kernel_height, kernel_width]
    weight_pt = pt_conv.weight.unsqueeze(2)  # [out_C, in_C, 1, kernel_size]
    weight_tt = ttnn.from_torch(weight_pt, dtype=ttnn.bfloat16)

    bias_pt = pt_conv.bias  # [out_C]
    bias_tt = ttnn.from_torch(bias_pt.view(1, 1, 1, -1), dtype=ttnn.bfloat16)

    # Run TTNN Conv1d
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        config_tensors_in_dram=True,
    )

    # Actually, ttnn.conv1d handles weight preprocessing if we just pass the original torch tensors
    # wait, the docstring says "It can be on host or device".
    # Let's pass the host weight tensor and let conv1d format it.

    try:
        y_tt, out_len, (tt_weight, tt_bias) = ttnn.conv1d(
            input_tensor=x_tt,
            weight_tensor=weight_tt,
            device=device,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            input_length=length,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias_tensor=bias_tt,
            dtype=ttnn.bfloat16,
            conv_config=conv_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
    except Exception as e:
        pytest.fail(f"TTNN conv1d failed: {e}")

    # Output should be [B, 1, L, C]
    y_tt_cpu = ttnn.to_torch(y_tt)
    # Convert to [B, C, L]
    y_tt_flat = y_tt_cpu.squeeze(1).transpose(1, 2)

    # Compare
    passing, pcc_msg = comp_pcc(y_pt, y_tt_flat, 0.99)
    print(f"PCC matching: {pcc_msg}")
    assert passing, f"PCC failed: {pcc_msg}"
