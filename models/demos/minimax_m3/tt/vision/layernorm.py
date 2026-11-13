# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
LayerNorm of the MiniMax-M3 vision tower, for pre_layrnorm and each layer's layer_norm1 and layer_norm2.
"""
from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimax_m3.tt.vision.common import hifi4_compute_config, to_device


class M3VLLayerNorm(LightweightModule):
    """LayerNorm with a learned weight and bias, built from the tensors or from an nn.LayerNorm (from_torch)."""

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

        # Weight and bias go to the device as [32, dim] tiles of copies of their row, as qwen3_vl's vision LayerNorm
        # lays them out; ttnn.layer_norm reads the first row of each tile.
        self.weight = to_device(weight.view(1, 1, dim).expand(1, ttnn.TILE_SIZE, dim), mesh_device, dtype)
        self.bias = to_device(bias.view(1, 1, dim).expand(1, ttnn.TILE_SIZE, dim), mesh_device, dtype)

        # Statistics in fp32: with fp32_dest_acc_en the op reduces in fp32 and keeps the mean, the centred row and the
        # variance in fp32 tiles.
        self.compute_kernel_config = hifi4_compute_config()

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        ref: torch.nn.LayerNorm,
        dtype=ttnn.bfloat16,
    ) -> "M3VLLayerNorm":
        """Build from an nn.LayerNorm: its weight, bias and eps.

        Raises NotImplementedError for a LayerNorm over more than the last dim.
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
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(
            x,
            weight=self.weight,
            bias=self.bias,
            epsilon=self.eps,
            compute_kernel_config=self.compute_kernel_config,
        )
