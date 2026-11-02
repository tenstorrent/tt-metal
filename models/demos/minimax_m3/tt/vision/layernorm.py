# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
LayerNorm wrapper for MiniMax-M3-VL blocks.

M3-VL (CLIP-style vision tower) uses standard nn.LayerNorm (mean subtraction
+ bias, eps=1e-5) at:
  - vision_tower.pre_layrnorm (after patch embed, before the encoder).
  - encoder layer .layer_norm1 and .layer_norm2 (pre-attn and pre-MLP).

All operate on the 1280-dim hidden state, so this single module covers every
LayerNorm site in the tower. Thin shim around ttnn.layer_norm (interleaved).
"""
from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimax_m3.tt.vision.common import hifi4_compute_config, to_device

TILE = ttnn.TILE_SIZE  # 32


class M3VLLayerNorm(LightweightModule):
    """Interleaved LayerNorm with learned weight + bias.

    Construct it from the weight and bias tensors, or with from_torch from an
    nn.LayerNorm, such as those of the Torch reference (reference/vision.py).
    """

    def __init__(
        self,
        mesh_device,
        dim: int,
        weight: torch.Tensor,
        bias: torch.Tensor,
        dtype=ttnn.bfloat16,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.eps = float(eps)

        assert (
            weight.ndim == 1 and weight.shape[0] == dim
        ), f"weight must be 1D of size {dim}, got {tuple(weight.shape)}"
        assert bias.ndim == 1 and bias.shape[0] == dim, f"bias must be 1D of size {dim}, got {tuple(bias.shape)}"

        # ttnn.layer_norm expects weight/bias laid out as TILE-aligned 2D.
        # Match the qwen3_vl convention: expand to [TILE, dim] so the
        # underlying op can broadcast across the height dim of any
        # input.
        self.weight = to_device(weight.view(1, 1, dim).expand(1, TILE, dim), mesh_device, dtype)
        self.bias = to_device(bias.view(1, 1, dim).expand(1, TILE, dim), mesh_device, dtype)

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        ref: torch.nn.LayerNorm,
        dtype=ttnn.bfloat16,
    ) -> "M3VLLayerNorm":
        """Construct from a torch.nn.LayerNorm reference module: its weight, bias, eps and normalized_shape[-1]."""
        assert isinstance(ref, torch.nn.LayerNorm), f"expected nn.LayerNorm, got {type(ref).__name__}"
        # normalized_shape is a tuple; LayerNorm normalizes over the last
        # `len(normalized_shape)` dims. For M3-VL it's always a 1-tuple (1280).
        if len(ref.normalized_shape) != 1:
            raise NotImplementedError(f"M3VLLayerNorm only supports 1-d normalized_shape, got {ref.normalized_shape}")
        return cls(
            mesh_device=mesh_device,
            dim=ref.normalized_shape[0],
            weight=ref.weight.data,
            bias=ref.bias.data,
            dtype=dtype,
            eps=ref.eps,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(
            x,
            weight=self.weight,
            bias=self.bias,
            epsilon=self.eps,
            compute_kernel_config=hifi4_compute_config(fp32_dest_acc=False),
        )
