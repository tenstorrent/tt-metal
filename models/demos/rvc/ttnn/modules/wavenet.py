# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN WaveNet (WN) module for RVC VITS.

Architecture reference (synthesizer.py L297-360):
    WN: num_layers × (dilated_conv → tanh*sigmoid → res_skip_linear → residual + skip)

Data flow for single WN layer:
    x[B,T,C] → conv1d(dilation=d) → [B,T,2C] → split → tanh*sigmoid → [B,T,C]
    → linear → split(res, skip) → residual, accumulate skip

Stage 1 design decisions:
    - Conv1d on device, gating channel split on host (torch fallback)
    - Conv bias added manually on host (not via ttnn.conv1d bias)
    - res_skip_linear uses ttnn.linear in channels-last format
    - Will optimize channel splitting with ttnn.slice in Stage 2

Validated: WN 3-layer PCC=0.999820
"""

import torch
import ttnn

from models.demos.rvc.ttnn.utils import (
    preprocess_conv1d_weight,
    preprocess_linear_weight,
    preprocess_linear_bias,
    DEFAULT_DTYPE,
)


def preprocess_wn_weights(
    in_layers: list,
    res_skip_layers: list,
    hidden_ch: int,
    num_layers: int,
    device,
) -> dict:
    """
    Preprocess all WN layer weights.

    Returns dict with conv_ws, conv_bs (host), rsl_ws, rsl_bs (device).
    """
    conv_ws = [preprocess_conv1d_weight(in_layers[i].weight.data) for i in range(num_layers)]
    conv_bs = [in_layers[i].bias.data.float() for i in range(num_layers)]
    rsl_ws = [preprocess_linear_weight(res_skip_layers[i].weight, device) for i in range(num_layers)]
    rsl_bs = [preprocess_linear_bias(res_skip_layers[i].bias, device) for i in range(num_layers)]

    return {
        "conv_ws": conv_ws,
        "conv_bs": conv_bs,
        "rsl_ws": rsl_ws,
        "rsl_bs": rsl_bs,
    }


def ttnn_wn_layer_forward(
    x_cl: torch.Tensor,
    conv_weight: ttnn.Tensor,
    conv_bias: torch.Tensor,
    rsl_weight: ttnn.Tensor,
    rsl_bias: ttnn.Tensor,
    hidden_ch: int,
    kernel_size: int,
    dilation: int,
    seq_len: int,
    device,
    is_last_layer: bool,
):
    """
    Single WN layer on TTNN.

    Args:
        x_cl: [B, T, C] channels-last tensor (host torch).
        conv_weight: Conv1d weight (host, ROW_MAJOR).
        conv_bias: Conv1d bias (host torch).
        rsl_weight: res_skip linear weight (device, TILE).
        rsl_bias: res_skip linear bias (device, TILE).
        hidden_ch: Hidden channels.
        kernel_size: Conv kernel size.
        dilation: Dilation rate.
        seq_len: Sequence length.
        device: TTNN device.
        is_last_layer: If True, all output goes to skip (no residual split).

    Returns:
        (x_cl_new, skip_cl): Both as host torch tensors [B, T, C].
    """
    padding = dilation * (kernel_size - 1) // 2

    # Dilated Conv1d
    x_tt = ttnn.from_torch(x_cl, dtype=DEFAULT_DTYPE)
    result = ttnn.conv1d(
        input_tensor=x_tt, weight_tensor=conv_weight, device=device,
        in_channels=hidden_ch, out_channels=2 * hidden_ch, batch_size=1,
        input_length=seq_len, kernel_size=kernel_size, stride=1,
        padding=padding, dilation=dilation, groups=1,
        dtype=DEFAULT_DTYPE, return_output_dim=True,
    )
    conv_out_tt, out_len = result[0], result[1]

    try:
        conv_out_tt = ttnn.sharded_to_interleaved(conv_out_tt)
    except RuntimeError:
        pass
    conv_torch = ttnn.to_torch(ttnn.from_device(conv_out_tt)).float()
    conv_torch = conv_torch.reshape(1, 1, out_len, -1)[:, :, :, :2*hidden_ch].squeeze(1)
    conv_torch = conv_torch + conv_bias.unsqueeze(0).unsqueeze(0)

    # Gating: tanh × sigmoid (channel split on host — Stage 1 pattern)
    tanh_in = conv_torch[:, :, :hidden_ch]
    sig_in = conv_torch[:, :, hidden_ch:]

    tanh_tt = ttnn.from_torch(tanh_in, dtype=DEFAULT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    sig_tt = ttnn.from_torch(sig_in, dtype=DEFAULT_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    gated = ttnn.mul(ttnn.tanh(tanh_tt), ttnn.sigmoid(sig_tt))

    # res_skip linear
    rs_out = ttnn.linear(gated, rsl_weight, bias=rsl_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rs_torch = ttnn.to_torch(rs_out).float()[:1, :seq_len, :]

    if not is_last_layer:
        return x_cl + rs_torch[:, :, :hidden_ch], rs_torch[:, :, hidden_ch:]
    else:
        return x_cl, rs_torch[:, :, :hidden_ch]


def ttnn_wn_forward(
    x_cl: torch.Tensor,
    wn_weights: dict,
    hidden_ch: int,
    kernel_size: int,
    dilation_rate: int,
    num_layers: int,
    seq_len: int,
    device,
) -> torch.Tensor:
    """
    Full WN stack forward pass.

    Args:
        x_cl: [B, T, hidden_ch] channels-last input.
        wn_weights: Dict from preprocess_wn_weights.
        hidden_ch: Hidden channels.
        kernel_size: Conv kernel size.
        dilation_rate: Base dilation rate (dilation = dilation_rate**i).
        num_layers: Number of WN layers.
        seq_len: Sequence length.
        device: TTNN device.

    Returns:
        [B, T, hidden_ch] accumulated skip output.
    """
    output_acc = torch.zeros(1, seq_len, hidden_ch)
    wn_x = x_cl

    for i in range(num_layers):
        d = dilation_rate ** i
        is_last = (i == num_layers - 1)

        wn_x, skip = ttnn_wn_layer_forward(
            wn_x,
            wn_weights["conv_ws"][i],
            wn_weights["conv_bs"][i],
            wn_weights["rsl_ws"][i],
            wn_weights["rsl_bs"][i],
            hidden_ch, kernel_size, d, seq_len, device, is_last,
        )
        output_acc = output_acc + skip

    return output_acc
