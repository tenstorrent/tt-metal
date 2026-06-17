# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC check for the hand-rolled depthwise causal conv1d + SiLU in my_gdn/operations.py.

Compares our FIR-decomposed TT-NN op against the exact HF expression
    F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
for a depthwise (groups=D), bias-free conv with left padding K-1 — i.e. the
no-cache prefill conv path of Qwen3.5's gated-delta-net.

Run from the repo root:  python tob.py
"""
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen3_5_9b.tt.my_gdn.operations import (
    causal_conv1d_silu,
    conv1d_weight_taps,
)

# Stand-in GDN conv dims: D ~ conv_dim (key_dim*2 + value_dim), K = linear_conv_kernel_dim.
B, T, D, K = 1, 128, 512, 4
PCC_TARGET = 0.99

torch.manual_seed(0)
device = ttnn.open_device(device_id=0)

try:
    # --- HF reference ---------------------------------------------------------
    # HF feeds mixed_qkv channels-first [B, D, T] (torch Conv1d's required layout).
    mixed_qkv = torch.randn(B, D, T)
    conv_weight = torch.randn(D, 1, K)  # depthwise: one [1, K] kernel per channel

    ref = F.silu(F.conv1d(mixed_qkv, conv_weight, bias=None, padding=K - 1, groups=D)[:, :, :T])  # [B, D, T]

    # --- our TT-NN op ---------------------------------------------------------
    # Our op is channels-last [B, T, D]; transpose in, run, transpose the result
    # back so we can line it up against HF's [B, D, T] directly.
    x = ttnn.from_torch(
        mixed_qkv.transpose(1, 2).contiguous(),  # [B, T, D]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    taps = conv1d_weight_taps(conv_weight, K, device)
    y = causal_conv1d_silu(x, taps, K, device)
    got = ttnn.to_torch(y).transpose(1, 2).contiguous()  # back to [B, D, T]

    # --- compare --------------------------------------------------------------
    passing, msg = comp_pcc(ref, got, pcc=PCC_TARGET)
    logger.info(f"conv1d+silu FIR vs HF: {msg}")
    print("PASS" if passing else "FAIL", "-", msg)
    assert passing, msg
finally:
    ttnn.close_device(device)
