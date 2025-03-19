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
import os


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
        ## for torch.unfold
        # weight = state["weight"].flatten(1, 3)
        # assert (weight.shape[-1] % 32) == 0
        # weight = weight.permute(1, 0).reshape(1, 1, -1, out_channels)
        # print("w_mod", weight.shape)
        ## for ttnn.fold
        weight = state["weight"]
        out_channels, in_c, kh, kw = weight.shape
        weight = torch.permute(weight, (2, 3, 1, 0))
        weight = torch.reshape(weight, (kh * kw * in_c, out_channels))

        if "bias" in state:
            bias = state["bias"].unsqueeze(0)
        else:
            bias = None

        if os.environ["FAKE_DEVICE"] == "T3K":
            hidden_dim = 2432
            hidden_dim_pad = 128
            hidden_dim_new = 2560
            weight_h, weight_w = weight.shape
            weight_w_mult = weight_w // hidden_dim
            if weight_w % hidden_dim == 0:
                if weight_w_mult == 1:
                    weight = torch.nn.functional.pad(weight, pad=(0, hidden_dim_pad), mode="constant", value=0)
                elif weight_w_mult > 1:
                    weight = weight.reshape(weight_h, weight_w_mult, hidden_dim)
                    weight = torch.nn.functional.pad(weight, pad=(0, hidden_dim_pad), mode="constant", value=0)
                    weight = weight.reshape(weight_h, weight_w_mult * hidden_dim_new)
            if not bias == None:
                bias_w = bias.shape[-1]
                bias_w_mult = bias_w // hidden_dim
                if bias_w % hidden_dim == 0:
                    if bias_w_mult == 1:
                        bias = torch.nn.functional.pad(bias, pad=(0, hidden_dim_pad), mode="constant", value=0)
                    elif bias_w_mult > 1:
                        bias = bias.reshape(bias_w_mult, hidden_dim)
                        bias = torch.nn.functional.pad(bias, pad=(0, hidden_dim_pad), mode="constant", value=0)
                        bias = bias.reshape(bias_w_mult * hidden_dim_new)

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
                    bias.reshape((1, 1, 1, -1)),
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
        batch_size, img_h, img_w, img_c = x.shape  # permuted input NHWC
        patch_size = 2
        stride_h = patch_size
        stride_w = 1

        unfolded_permuted_x = ttnn.fold(x, stride_h, stride_w)
        folded_shape = unfolded_permuted_x.shape
        unfolded_permuted_x = ttnn.to_layout(unfolded_permuted_x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

        ttnn.deallocate(x)

        out = ttnn.linear(
            unfolded_permuted_x,
            self._weight,
            bias=self._bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )
        ttnn.deallocate(unfolded_permuted_x)
        seq_len = out.shape[-2] // batch_size
        out = ttnn.reshape(out, (batch_size, seq_len, -1))

        return out
