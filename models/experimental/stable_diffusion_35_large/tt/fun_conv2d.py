# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import ttnn

from .parallel_config import DiTParallelConfig
from .utils import from_torch_fast


@dataclass
class TtConv2dParameters:
    weight: ttnn.Tensor
    bias: ttnn.Tensor | None
    out_channels: int
    kernel_size: tuple[int, int]

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        hidden_dim_padding: int,
        out_channels: int,
        device,
        parallel_config: DiTParallelConfig,
    ) -> TtConv2dParameters:
        # TODO: Use parallel_config
        weight = state["weight"]
        out_channels, in_c, kh, kw = weight.shape
        weight = torch.permute(weight, (2, 3, 1, 0))
        weight = torch.reshape(weight, (kh * kw * in_c, out_channels))

        if "bias" in state:
            bias = state["bias"].unsqueeze(0)
        else:
            bias = None

        if os.environ["FAKE_DEVICE"] == "T3K":
            weight = torch.nn.functional.pad(weight, pad=(0, hidden_dim_padding), mode="constant", value=0)
            if not bias == None:
                bias = torch.nn.functional.pad(bias, pad=(0, hidden_dim_padding), mode="constant", value=0)

        return cls(
            weight=from_torch_fast(
                weight,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                shard_dim=-1,
            ),
            bias=(
                from_torch_fast(
                    bias.reshape((1, 1, 1, -1)),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    shard_dim=-1,
                )
                if "bias" in state
                else None
            ),
            out_channels=out_channels,
            kernel_size=(kh, kw),
        )


def sd_conv2d(x: ttnn.Tensor, parameters: TtConv2dParameters, parallel_config: DiTParallelConfig) -> ttnn.Tensor:
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        x.device().arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    batch_size, img_h, img_w, img_c = x.shape  # permuted input NHWC
    patch_size = 2
    stride_h = patch_size
    stride_w = 1
    patches_h = img_h // patch_size
    patches_w = img_w // patch_size

    x = ttnn.reshape(x, (batch_size, patches_h, patch_size, patches_w, patch_size, img_c))
    x = ttnn.permute(x, (0, 1, 3, 2, 4, 5))
    x = ttnn.reshape(x, (batch_size, patches_h * patches_w, patch_size * patch_size * img_c))
    out = ttnn.linear(
        x,
        parameters.weight,
        bias=parameters.bias,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_kernel_config,
        core_grid=ttnn.CoreGrid(y=8, x=8),
    )
    out = ttnn.reshape(out, (batch_size, patches_h, patches_w, -1))
    return out
