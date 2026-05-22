# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
VibeVoice SpeechConnector — TTNN port.

Reference: SpeechConnector in modeling_vibevoice.py lines 57–68
  fc1 → LlamaRMSNorm(eps=1e-6) → fc2

Allowed imports: ttnn, dataclasses, typing, math
Host-side preprocessing (preprocess_connector_parameters) may use torch.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import ttnn


@dataclass
class ConnectorParameters:
    fc1_weight: ttnn.Tensor  # [out, in]
    fc2_weight: ttnn.Tensor  # [out, in]
    norm_weight: ttnn.Tensor  # [1, 1, 1, hidden_size]
    fc1_bias: Optional[ttnn.Tensor] = None  # [1, 1, 1, hidden]
    fc2_bias: Optional[ttnn.Tensor] = None  # [1, 1, 1, hidden]
    input_dim: int = 0
    hidden_dim: int = 0
    eps: float = 1e-6


def preprocess_connector_parameters(
    hf_state: dict,
    device,
    eps: float = 1e-6,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ConnectorParameters:
    """Convert host-side HF SpeechConnector weights to TTNN tensors.

    hf_state keys expected: fc1.weight, fc1.bias (opt), fc2.weight, fc2.bias (opt), norm.weight
    (stripped of module prefix by split_submodule_weights).
    """
    fc1_w = hf_state["fc1.weight"].to(torch.bfloat16)  # [hidden, input]
    fc2_w = hf_state["fc2.weight"].to(torch.bfloat16)  # [hidden, hidden]
    norm_w = hf_state["norm.weight"].to(torch.bfloat16)  # [hidden]

    hidden_dim = fc1_w.shape[0]
    input_dim = fc1_w.shape[1]

    # ttnn.linear computes x @ W (no implicit transpose), so store weights as [in, out]
    fc1_tt = ttnn.as_tensor(
        fc1_w.t().unsqueeze(0).unsqueeze(0),  # [1, 1, in, out]
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    fc2_tt = ttnn.as_tensor(
        fc2_w.t().unsqueeze(0).unsqueeze(0),  # [1, 1, in, out]
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # RMSNorm weight: ttnn.rms_norm requires shape [1, 1, hidden//32, 32] in ROW_MAJOR
    norm_w_4d = norm_w.view(1, 1, hidden_dim // 32, 32)
    norm_tt = ttnn.as_tensor(
        norm_w_4d,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def _bias_to_tt(b):
        return ttnn.as_tensor(
            b.to(torch.bfloat16).view(1, 1, 1, -1),
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    fc1_b_raw = hf_state.get("fc1.bias")
    fc2_b_raw = hf_state.get("fc2.bias")
    fc1_b_tt = _bias_to_tt(fc1_b_raw) if fc1_b_raw is not None else None
    fc2_b_tt = _bias_to_tt(fc2_b_raw) if fc2_b_raw is not None else None

    return ConnectorParameters(
        fc1_weight=fc1_tt,
        fc2_weight=fc2_tt,
        norm_weight=norm_tt,
        fc1_bias=fc1_b_tt,
        fc2_bias=fc2_b_tt,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        eps=eps,
    )


_COMPUTE_KERNEL_HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


class TTSpeechConnector:
    """TTNN port of SpeechConnector: fc1 → RMSNorm → fc2.

    No torch in forward. All operations on device tensors.
    """

    def __init__(self, params: ConnectorParameters):
        self.params = params

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            x: ttnn.Tensor [B, T, input_dim] in bfloat16, TILE layout

        Returns:
            ttnn.Tensor [B, T, hidden_dim]
        """
        p = self.params

        # fc1: [B, T, input_dim] → [B, T, hidden_dim]
        x = ttnn.linear(
            x,
            p.fc1_weight,
            bias=p.fc1_bias,
            compute_kernel_config=_COMPUTE_KERNEL_HIFI4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # RMSNorm with eps=1e-6
        x = ttnn.rms_norm(
            x,
            weight=p.norm_weight,
            epsilon=p.eps,
            compute_kernel_config=_COMPUTE_KERNEL_HIFI4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # fc2: [B, T, hidden_dim] → [B, T, hidden_dim]
        x = ttnn.linear(
            x,
            p.fc2_weight,
            bias=p.fc2_bias,
            compute_kernel_config=_COMPUTE_KERNEL_HIFI4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return x

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.forward(x)
