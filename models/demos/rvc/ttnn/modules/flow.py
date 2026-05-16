# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Flow Decoder module for RVC VITS.

Architecture reference (synthesizer.py L417-489):
    ResidualCouplingLayer:
        split x → x0, x1
        h = pre_linear(x0) → WN(h) → post_linear(h)
        x1 = x1 - h
        x = cat(x0, x1)

    ResidualCouplingBlock:
        for each flow: flip(x) → ResidualCouplingLayer(x)

RVC config: channels=192, hidden=192, kernel=5, dilation_rate=1, num_layers=3, n_flows=4

Stage 1 design decisions:
    - Channel flip on host (torch.flip)
    - Channel split on host (slicing)
    - pre/post linear on device (ttnn.linear)
    - WN on device (via wavenet module)
    - All concat on host

Validated: 4-flow PCC=0.999995
"""

import torch
import ttnn

from models.demos.rvc.ttnn.utils import (
    preprocess_linear_weight,
    preprocess_linear_bias,
    DEFAULT_DTYPE,
)
from models.demos.rvc.ttnn.modules.wavenet import (
    preprocess_wn_weights,
    ttnn_wn_forward,
)


def preprocess_flow_weights(
    n_flows: int,
    hidden_ch: int,
    kernel_size: int,
    dilation_rate: int,
    num_layers: int,
    torch_layers: list,
    device,
) -> list:
    """
    Preprocess all flow decoder weights.

    Args:
        n_flows: Number of flow steps.
        hidden_ch: WN hidden channels.
        kernel_size: WN conv kernel size.
        dilation_rate: WN dilation base.
        num_layers: WN layers per flow.
        torch_layers: List of (pre_linear, post_linear, in_layers, res_skip_layers).
        device: TTNN device.

    Returns:
        List of flow weight dicts.
    """
    all_flow = []
    for f in range(n_flows):
        pre_l, post_l, ins, rsls = torch_layers[f]

        pre = {
            "weight": preprocess_linear_weight(pre_l.weight, device),
            "bias": preprocess_linear_bias(pre_l.bias, device),
        }
        post = {
            "weight": preprocess_linear_weight(post_l.weight, device),
            "bias": preprocess_linear_bias(post_l.bias, device),
        }
        wn = preprocess_wn_weights(ins, rsls, hidden_ch, num_layers, device)

        all_flow.append({"pre": pre, "post": post, "wn": wn})
    return all_flow


def ttnn_flow_forward(
    x_nchw: torch.Tensor,
    flow_weights: list,
    hidden_ch: int,
    kernel_size: int,
    dilation_rate: int,
    num_layers: int,
    n_flows: int,
    seq_len: int,
    device,
) -> torch.Tensor:
    """
    Full flow decoder forward pass.

    Args:
        x_nchw: Input [B, C, T] (channels-first, PyTorch format).
        flow_weights: List from preprocess_flow_weights.
        hidden_ch: WN hidden channels.
        kernel_size: WN conv kernel size.
        dilation_rate: WN dilation base.
        num_layers: WN layers per flow.
        n_flows: Number of flow steps.
        seq_len: Sequence length.
        device: TTNN device.

    Returns:
        [B, C, T] output (channels-first).
    """
    channels = x_nchw.shape[1]
    half = channels // 2

    x_cl = x_nchw.permute(0, 2, 1)  # [B, T, C]

    for f in range(n_flows):
        # Channel flip (host, Stage 1)
        x_cl = torch.flip(x_cl, [2])

        fw = flow_weights[f]

        # Channel split (host)
        x0_cl = x_cl[:, :, :half]
        x1_cl = x_cl[:, :, half:]

        # pre_linear
        x0_tt = ttnn.from_torch(x0_cl, dtype=DEFAULT_DTYPE,
                                 layout=ttnn.TILE_LAYOUT, device=device)
        h_tt = ttnn.linear(x0_tt, fw["pre"]["weight"], bias=fw["pre"]["bias"],
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h_cl = ttnn.to_torch(h_tt).float()[:1, :seq_len, :hidden_ch]

        # WN
        output_acc = ttnn_wn_forward(
            h_cl, fw["wn"], hidden_ch, kernel_size,
            dilation_rate, num_layers, seq_len, device)

        # post_linear
        wn_out_tt = ttnn.from_torch(output_acc, dtype=DEFAULT_DTYPE,
                                     layout=ttnn.TILE_LAYOUT, device=device)
        stats_tt = ttnn.linear(wn_out_tt, fw["post"]["weight"], bias=fw["post"]["bias"],
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
        stats_cl = ttnn.to_torch(stats_tt).float()[:1, :seq_len, :half]

        # Subtract + concat (host)
        x1_cl = x1_cl - stats_cl
        x_cl = torch.cat([x0_cl, x1_cl], dim=-1)

    return x_cl.permute(0, 2, 1)  # [B, C, T]
