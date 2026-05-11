# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Kokoro `bert_encoder` (Linear after PL-BERT).

Full ALBERT on TT is tracked separately; this module maps the projection
`hidden_size -> cfg["hidden_dim"]` used by the predictor stack.
"""

from __future__ import annotations

from typing import Optional

import ttnn
from models.experimental.kokoro.tt.common import default_compute_kernel_config, linear_output_memory_config


def _reshape_linear_out_to_btc(
    x: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig,
) -> ttnn.Tensor:
    """Normalize linear output to rank-3 [batch, seq, features] (BH may yield rank 4)."""
    shape = list(x.shape)
    rank = len(shape)
    if rank == 3:
        return x
    if rank == 4:
        # Expected [B, 1, T, out] when an extra singleton appears after matmul.
        if shape[1] == 1:
            return ttnn.reshape(x, [shape[0], shape[2], shape[3]], memory_config=memory_config)
        # [B, T, out, 1] or similar — collapse trailing singleton
        if shape[-1] == 1:
            return ttnn.reshape(x, [shape[0], shape[1], shape[2]], memory_config=memory_config)
    raise ValueError(f"Unexpected linear output rank {rank}, shape={shape}")


class TtKokoroPlBertProjection:
    """Applies the Kokoro `bert_encoder` linear on device."""

    def __init__(
        self,
        mesh_device,
        *,
        parameters: dict[str, ttnn.Tensor],
        compute_kernel_config=None,
    ):
        self.mesh_device = mesh_device
        self.weight = parameters["weight"]
        self.bias = parameters.get("bias")
        self.compute_kernel_config = compute_kernel_config or default_compute_kernel_config(mesh_device)

    def __call__(
        self,
        bert_dur: ttnn.Tensor,
        *,
        output_memory_config: Optional[ttnn.MemoryConfig] = None,
    ) -> ttnn.Tensor:
        """
        Args:
            bert_dur: [batch, seq, hidden_size] (same layout as ALBERT `last_hidden_state`).

        Returns:
            d_en: [batch, hidden_dim, seq] matching reference `transpose(-1, -2)` after Linear.
        """
        mem_cfg = output_memory_config or linear_output_memory_config(self.mesh_device)
        # nn.Linear weight is [out_features, in_features]; ttnn.linear needs transpose_b=True
        # so the matmul matches torch.nn.functional.linear(activations, weight, bias).
        hidden = ttnn.linear(
            bert_dur,
            self.weight,
            bias=self.bias,
            transpose_b=True,
            memory_config=mem_cfg,
            compute_kernel_config=self.compute_kernel_config,
        )
        hidden = _reshape_linear_out_to_btc(hidden, memory_config=mem_cfg)
        # [B, T, out] -> [B, out, T] (prefer transpose over permute for rank stability)
        return ttnn.transpose(hidden, 1, 2, memory_config=mem_cfg)
