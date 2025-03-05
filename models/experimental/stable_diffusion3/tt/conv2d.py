# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn
from models.utility_functions import (
    nearest_32,
)

import torch


@dataclass
class TtConv2dParameters:
    weight: ttnn.Tensor
    bias: ttnn.Tensor | None
    out_channels: int
    kernel_size: int

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        out_channels: int,
        device,
    ) -> TtConv2dParameters:
        weight = state["weight"].flatten(1, 3)
        assert (weight.shape[-1] % 32) == 0
        # pad_len = nearest_32(weight.shape[-1]) - weight.shape[-1]
        # padding = torch.zeros(out_channels, pad_len, dtype=weight.dtype)
        # padded_weight = torch.cat([weight, padding], dim=-1)
        weight = weight.permute(1, 0).reshape(1, 1, -1, out_channels)

        return cls(
            weight=ttnn.as_tensor(
                weight,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            ),
            bias=(
                ttnn.as_tensor(
                    state["bias"].reshape((1, 1, 1, -1)),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(device),
                )
                if "bias" in state
                else None
            ),
            out_channels=out_channels,
            kernel_size=state["weight"].shape[-1],
        )


class TtConv2d:
    def __init__(self, parameters: TtConv2dParameters, device) -> None:
        self._kernel_size = parameters.kernel_size

        self._weight = parameters.weight
        self._bias = parameters.bias
        self._unfold = torch.nn.Unfold(kernel_size=(self._kernel_size,) * 2, stride=(self._kernel_size,) * 2)
        self._device = device

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        host_x = ttnn.from_device(x)
        torch_tensor = ttnn.to_torch(host_x, mesh_composer=ttnn.ConcatMeshToTensor(self._device, dim=0)).permute(
            [0, 3, 1, 2]
        )
        unfolded_x = ttnn.as_tensor(
            self._unfold(torch_tensor[0 : x.shape[0], ...]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self._device),
        )
        unfolded_permuted_x = ttnn.permute(unfolded_x, (0, 2, 1))

        # Need to pad the last dimension of x to be a multiple of a tile
        assert (unfolded_permuted_x.shape[-1] % 32) == 0
        # pad_len = nearest_32(x.shape[-1]) - x.shape[-1]
        # padding = torch.zeros((x.shape[0], x.shape[1], pad_len), dtype=x.dtype, device=x.device)
        # x = torch.cat([x, padding], dim=-1)

        out = ttnn.linear(
            unfolded_permuted_x,
            self._weight,
            bias=self._bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )

        return out
