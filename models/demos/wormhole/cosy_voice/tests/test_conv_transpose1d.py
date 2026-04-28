import pytest
import torch
import torch.nn as nn

import ttnn
from models.common.utility_functions import comp_pcc


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, stride, padding",
    [
        (256, 128, 16, 8, 4),  # Example from Vocoder upsample
    ],
)
@pytest.mark.parametrize("length", [16])  # Smaller length for quick testing
def test_conv_transpose1d(device, in_channels, out_channels, kernel_size, stride, padding, length):
    torch.manual_seed(0)
    batch_size = 1

    # PyTorch Setup
    pt_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    # Input for PyTorch: [B, C, L]
    x_pt = torch.randn(batch_size, in_channels, length)
    y_pt = pt_conv(x_pt)

    # TTNN Setup
    # TTNN expects input in [N, H, W, C] -> [Batch, 1, Length, Channels]
    x_tt = x_pt.transpose(1, 2).unsqueeze(1)  # [B, 1, L, C]
    x_tt = ttnn.from_torch(x_tt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Weights for ConvTranspose1d: [in_channels, out_channels, kernel_size]
    # For ttnn.conv_transpose2d, we reshape it to [in_channels, out_channels, 1, kernel_size]
    weight_pt = pt_conv.weight.unsqueeze(2)  # [in_C, out_C, 1, kernel_size]
    weight_tt = ttnn.from_torch(weight_pt, dtype=ttnn.bfloat16)

    bias_pt = pt_conv.bias  # [out_C]
    bias_tt = ttnn.from_torch(bias_pt.view(1, 1, 1, -1), dtype=ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        config_tensors_in_dram=True,
    )

    try:
        y_tt, out_hw, (tt_weight, tt_bias) = ttnn.conv_transpose2d(
            input_tensor=x_tt,
            weight_tensor=weight_tt,
            device=device,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            input_height=1,
            input_width=length,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, padding),
            output_padding=(0, 0),
            groups=1,
            bias_tensor=bias_tt,
            conv_config=conv_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
    except Exception as e:
        pytest.fail(f"TTNN conv_transpose2d failed: {e}")

    # Output should be [B, 1, L_out, C]
    y_tt_cpu = ttnn.to_torch(y_tt)
    # Convert to [B, C, L_out]
    y_tt_flat = y_tt_cpu.squeeze(1).transpose(1, 2)

    # Note: ttnn output might have padded dims, so we might need to truncate
    expected_length = y_pt.shape[2]
    y_tt_flat = y_tt_flat[:, :, :expected_length]

    # Compare
    passing, pcc_msg = comp_pcc(y_pt, y_tt_flat, 0.99)
    print(f"PCC matching: {pcc_msg}")
    assert passing, f"PCC failed: {pcc_msg}"
