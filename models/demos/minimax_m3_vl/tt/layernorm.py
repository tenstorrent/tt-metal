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

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimax_m3_vl.tt.common import hifi4_compute_config, mesh_mapper

TILE = ttnn.TILE_SIZE  # 32


class M3VLLayerNorm(LightweightModule):
    """Interleaved LayerNorm with learned weight + bias.

    Construct either by passing the torch weight/bias tensors directly
    (preferred for tests — the HF reference module already exposes
    them on its state_dict) or via the state_dict + key pattern used
    elsewhere in tt-metal.
    """

    def __init__(
        self,
        mesh_device,
        dim: int,
        weight: torch.Tensor,
        bias: torch.Tensor,
        dtype=ttnn.bfloat16,
        eps: float = 1e-5,
        weight_memory_config=None,
    ):
        super().__init__()
        self.device = mesh_device
        self.eps = float(eps)
        self.dtype = dtype
        self.dim = int(dim)

        assert (
            weight.ndim == 1 and weight.shape[0] == self.dim
        ), f"weight must be 1D of size {self.dim}, got {tuple(weight.shape)}"
        assert (
            bias.ndim == 1 and bias.shape[0] == self.dim
        ), f"bias must be 1D of size {self.dim}, got {tuple(bias.shape)}"

        # ttnn.layer_norm expects weight/bias laid out as TILE-aligned 2D.
        # Match the qwen3_vl convention: expand to [TILE, dim] so the
        # underlying op can broadcast across the height dim of any
        # input. The expand is a view (no extra memory on host).
        torch_weight = weight.detach().to(torch.bfloat16).view(1, 1, self.dim).expand(1, TILE, self.dim).contiguous()
        torch_bias = bias.detach().to(torch.bfloat16).view(1, 1, self.dim).expand(1, TILE, self.dim).contiguous()

        mem_cfg = weight_memory_config if weight_memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
        mm = mesh_mapper(mesh_device)

        self.weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_cfg,
            mesh_mapper=mm,
        )
        self.bias = ttnn.as_tensor(
            torch_bias,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_cfg,
            mesh_mapper=mm,
        )

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        ref: torch.nn.LayerNorm,
        dtype=ttnn.bfloat16,
        weight_memory_config=None,
    ) -> "M3VLLayerNorm":
        """Construct from a torch.nn.LayerNorm reference module.

        Pulls out `ref.weight`, `ref.bias`, `ref.eps`, and `ref.normalized_shape[-1]`
        — i.e., the obvious mapping. Useful in tests where the
        comparison reference is the source of truth.
        """
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
            weight_memory_config=weight_memory_config,
        )

    def forward(self, x: ttnn.Tensor, memory_config: Optional["ttnn.MemoryConfig"] = None) -> ttnn.Tensor:
        return ttnn.layer_norm(
            x,
            weight=self.weight,
            bias=self.bias,
            epsilon=self.eps,
            memory_config=memory_config,
            compute_kernel_config=hifi4_compute_config(fp32_dest_acc=False),
        )

    # `__call__` is provided by LightweightModule and dispatches to forward.
