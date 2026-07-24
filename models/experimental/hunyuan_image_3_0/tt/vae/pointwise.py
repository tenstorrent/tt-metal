# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""1x1x1 Conv3d as ``ttnn.linear`` — avoids im2col-bound conv3d for pointwise ops."""

from __future__ import annotations

from typing import Any

import torch
import ttnn
from models.common.utility_functions import is_blackhole
from models.tt_dit.layers.module import Module, Parameter
from models.tt_dit.utils.conv3d import aligned_channels


class HunyuanPointwiseLinear(Module):
    """Pointwise (1x1x1) projection on BTHWC tensors via ``ttnn.linear``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.unpadded_in_channels = in_channels
        self.in_channels = aligned_channels(in_channels)
        self.out_channels = out_channels
        self.mesh_device = mesh_device
        self.dtype = dtype

        self.weight = Parameter(
            total_shape=[self.in_channels, out_channels],
            device=mesh_device,
            pad_value=0,
            dtype=dtype,
        )
        self.bias = Parameter(
            total_shape=[1, out_channels],
            device=mesh_device,
            pad_value=0,
            dtype=dtype,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2
            if (is_blackhole() and dtype == ttnn.float32)
            else ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, Any]) -> None:
        weight = state.get("weight")
        if weight is not None:
            if weight.ndim == 5:
                # Conv3d checkpoint: [out, in, 1, 1, 1] -> linear [in, out]
                weight = weight.squeeze(-1).squeeze(-1).squeeze(-1)
                weight = weight.transpose(0, 1).contiguous()
            elif tuple(weight.shape) == (self.in_channels, self.out_channels):
                # Already linear layout (e.g. fused QKV from load_attn_block).
                weight = weight.contiguous()
            elif weight.shape[0] == self.out_channels:
                weight = weight.transpose(0, 1).contiguous()
            if weight.shape[0] < self.in_channels:
                weight = torch.nn.functional.pad(weight, (0, 0, 0, self.in_channels - weight.shape[0]))
            state["weight"] = weight
        bias = state.get("bias")
        if bias is not None:
            state["bias"] = bias.reshape(1, -1)

    def forward(self, x_bthwc: ttnn.Tensor, *, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
        x_tile = ttnn.to_layout(x_bthwc, ttnn.TILE_LAYOUT)
        y_tile = ttnn.linear(
            x_tile,
            self.weight.data,
            bias=self.bias.data,
            compute_kernel_config=self.compute_kernel_config,
            dtype=self.dtype,
            memory_config=memory_config,
            core_grid=self.mesh_device.core_grid,
        )
        ttnn.deallocate(x_tile)
        y = ttnn.to_layout(y_tile, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(y_tile)
        return y

    def forward_b1sc(
        self,
        x_b1sc: ttnn.Tensor,
        *,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        """Linear on ``[B, 1, S, C]`` TILE input (last dim = channels)."""
        return ttnn.linear(
            x_b1sc,
            self.weight.data,
            bias=self.bias.data,
            compute_kernel_config=self.compute_kernel_config,
            dtype=self.dtype,
            memory_config=memory_config,
            core_grid=self.mesh_device.core_grid,
        )
