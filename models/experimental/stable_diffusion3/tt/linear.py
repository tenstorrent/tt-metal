# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from .utils import from_torch

# if TYPE_CHECKING:
import torch
import os


@dataclass
class TtLinearParameters:
    weight: ttnn.Tensor
    bias: ttnn.Tensor | None

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
        shard_dim: int,
    ) -> TtLinearParameters:
        if "bias" in state:
            bias = state["bias"].unsqueeze(0)
        else:
            bias = None
        weight = state["weight"]
        if os.environ["FAKE_DEVICE"] == "T3K":
            hidden_dim = 2432
            hidden_dim_pad = 128
            hidden_dim_new = 2560
            weight_h, weight_w = weight.shape
            weight_h_mult = weight_h // hidden_dim
            weight_w_mult = weight_w // hidden_dim
            if weight_h % hidden_dim == 0:
                if weight_h_mult == 1:
                    weight = torch.nn.functional.pad(weight, pad=(0, 0, 0, hidden_dim_pad), mode="constant", value=0)
                elif weight_h_mult > 1:
                    weight = weight.reshape(weight_h_mult, hidden_dim, weight_w)
                    weight = torch.nn.functional.pad(weight, pad=(0, 0, 0, hidden_dim_pad), mode="constant", value=0)
                    weight = weight.reshape(weight_h_mult * hidden_dim_new, weight_w)
                weight_h, weight_w = weight.shape
                if weight_w_mult == 1:
                    weight = torch.nn.functional.pad(weight, pad=(0, hidden_dim_pad), mode="constant", value=0)
                elif weight_w_mult > 1:
                    weight = weight.reshape(weight_h, weight_w_mult, hidden_dim)
                    weight = torch.nn.functional.pad(weight, pad=(0, hidden_dim_pad), mode="constant", value=0)
                    weight = weight.reshape(weight_h, weight_w_mult * hidden_dim_new)

                if not bias == None:
                    bias_h, bias_w = bias.shape
                    bias_w_mult = bias_w // hidden_dim
                    if bias_w_mult == 1:
                        bias = torch.nn.functional.pad(bias, pad=(0, hidden_dim_pad), mode="constant", value=0)
                    elif bias_w_mult > 1:
                        bias = bias.reshape(bias_h, bias_w_mult, hidden_dim)
                        bias = torch.nn.functional.pad(bias, pad=(0, hidden_dim_pad), mode="constant", value=0)
                        bias = bias.reshape(bias_h, bias_w_mult * hidden_dim_new)
        return cls(
            weight=from_torch(
                weight.transpose(0, 1),
                dtype=dtype,
                mesh_device=device,
                shard_dim=shard_dim,
                layout=ttnn.TILE_LAYOUT,
            ),
            bias=from_torch(
                bias,
                dtype=dtype,
                mesh_device=device,
                shard_dim=shard_dim,
                layout=ttnn.TILE_LAYOUT,
            )
            if bias is not None
            else None,
        )

    @classmethod
    def from_torch_col_parallel(
        cls,
        state: dict[str, torch.Tensor],
        *,
        n_local_heads: int,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtLinearParameters:
        if "bias" in state:
            torch_bias = state["bias"].unsqueeze(0)
        else:
            torch_bias = None

        weight = state["weight"]
        if os.environ["FAKE_DEVICE"] == "T3K":
            hidden_dim = 2432
            hidden_dim_pad = 128
            hidden_dim_new = 2560
            weight_h, weight_w = weight.shape
            weight_h_mult = weight_h // hidden_dim
            weight_w_mult = weight_w // hidden_dim
            if weight_h % hidden_dim == 0:
                if weight_h_mult == 1:
                    weight = torch.nn.functional.pad(weight, pad=(0, 0, 0, hidden_dim_pad), mode="constant", value=0)
                elif weight_h_mult > 1:
                    weight = weight.reshape(weight_h_mult, hidden_dim, weight_w)
                    weight = torch.nn.functional.pad(weight, pad=(0, 0, 0, hidden_dim_pad), mode="constant", value=0)
                    weight = weight.reshape(weight_h_mult * hidden_dim_new, weight_w)
                weight_h, weight_w = weight.shape
                if weight_w_mult == 1:
                    weight = torch.nn.functional.pad(weight, pad=(0, hidden_dim_pad), mode="constant", value=0)
                elif weight_w_mult > 1:
                    weight = weight.reshape(weight_h, weight_w_mult, hidden_dim)
                    weight = torch.nn.functional.pad(weight, pad=(0, hidden_dim_pad), mode="constant", value=0)
                    weight = weight.reshape(weight_h, weight_w_mult * hidden_dim_new)

                if not torch_bias == None:
                    bias_h, bias_w = torch_bias.shape
                    bias_w_mult = bias_w // hidden_dim
                    if bias_w_mult == 1:
                        torch_bias = torch.nn.functional.pad(
                            torch_bias, pad=(0, hidden_dim_pad), mode="constant", value=0
                        )
                    elif bias_w_mult > 1:
                        torch_bias = torch_bias.reshape(bias_h, bias_w_mult, hidden_dim)
                        torch_bias = torch.nn.functional.pad(
                            torch_bias, pad=(0, hidden_dim_pad), mode="constant", value=0
                        )
                        torch_bias = torch_bias.reshape(bias_h, bias_w_mult * hidden_dim_new)

        def shuffle_heads(tensor):
            # Given torch tensor with output features in the last dimension,
            # shuffle heads to allow for column parallel computation
            in_dim = tensor.shape[0]
            tensor = tensor.reshape(in_dim, 3, device.get_num_devices(), n_local_heads, -1)  # [ID, 3, ND, NLH, DH]
            tensor = tensor.permute(0, 2, 1, 3, 4)  # [ID, ND, 3, NLH, DH]
            tensor = tensor.reshape(in_dim, -1)  # [ID, ND*3*NLH*DH]
            return tensor

        torch_weight = weight.transpose(0, 1)
        return cls(
            weight=from_torch(
                shuffle_heads(torch_weight),
                dtype=dtype,
                mesh_device=device,
                shard_dim=-1,
                layout=ttnn.TILE_LAYOUT,
            ),
            bias=from_torch(
                shuffle_heads(torch_bias),
                dtype=dtype,
                mesh_device=device,
                shard_dim=-1,
                layout=ttnn.TILE_LAYOUT,
            )
            if torch_bias is not None
            else None,
        )

    @classmethod
    def from_torch_time_embed(
        cls,
        state: dict[str, torch.Tensor],
        *,
        num_chunks: int,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtLinearParameters:
        if "bias" in state:
            torch_bias = state["bias"].unsqueeze(0)
        else:
            torch_bias = None

        weight = state["weight"]
        if os.environ["FAKE_DEVICE"] == "T3K":
            hidden_dim = 2432
            hidden_dim_pad = 128
            hidden_dim_new = 2560
            weight_h, weight_w = weight.shape
            weight_h_mult = weight_h // hidden_dim
            weight_w_mult = weight_w // hidden_dim
            if weight_h % hidden_dim == 0:
                if weight_h_mult == 1:
                    weight = torch.nn.functional.pad(weight, pad=(0, 0, 0, hidden_dim_pad), mode="constant", value=0)
                elif weight_h_mult > 1:
                    weight = weight.reshape(weight_h_mult, hidden_dim, weight_w)
                    weight = torch.nn.functional.pad(weight, pad=(0, 0, 0, hidden_dim_pad), mode="constant", value=0)
                    weight = weight.reshape(weight_h_mult * hidden_dim_new, weight_w)
                weight_h, weight_w = weight.shape
                if weight_w_mult == 1:
                    weight = torch.nn.functional.pad(weight, pad=(0, hidden_dim_pad), mode="constant", value=0)
                elif weight_w_mult > 1:
                    weight = weight.reshape(weight_h, weight_w_mult, hidden_dim)
                    weight = torch.nn.functional.pad(weight, pad=(0, hidden_dim_pad), mode="constant", value=0)
                    weight = weight.reshape(weight_h, weight_w_mult * hidden_dim_new)

                if not torch_bias == None:
                    bias_h, bias_w = torch_bias.shape
                    bias_w_mult = bias_w // hidden_dim
                    if bias_w_mult == 1:
                        torch_bias = torch.nn.functional.pad(
                            torch_bias, pad=(0, hidden_dim_pad), mode="constant", value=0
                        )
                    elif bias_w_mult > 1:
                        torch_bias = torch_bias.reshape(bias_h, bias_w_mult, hidden_dim)
                        torch_bias = torch.nn.functional.pad(
                            torch_bias, pad=(0, hidden_dim_pad), mode="constant", value=0
                        )
                        torch_bias = torch_bias.reshape(bias_h, bias_w_mult * hidden_dim_new)

        def shuffle_chunks(tensor):
            # Given torch tensor with output features in the last dimension,
            # shuffle heads to allow for column parallel computation
            in_dim = tensor.shape[0]
            tensor = tensor.reshape(in_dim, num_chunks, device.get_num_devices(), -1)
            tensor = tensor.permute(0, 2, 1, 3)
            tensor = tensor.reshape(in_dim, -1)
            return tensor

        torch_weight = weight.transpose(0, 1)
        return cls(
            weight=from_torch(
                shuffle_chunks(torch_weight),
                dtype=dtype,
                mesh_device=device,
                shard_dim=-1,
            ),
            bias=from_torch(
                shuffle_chunks(torch_bias),
                dtype=dtype,
                mesh_device=device,
                shard_dim=-1,
            )
            if torch_bias is not None
            else None,
        )

    @property
    def in_channels(self) -> int:
        return self.weight.shape[0]

    @property
    def out_channels(self) -> int:
        return self.weight.shape[1]


class TtLinear:
    def __init__(self, parameters: TtLinearParameters) -> None:
        self._in_channels = parameters.in_channels
        self._weight = parameters.weight
        self._bias = parameters.bias

    def __call__(
        self,
        x: ttnn.Tensor,
        *,
        memory_config: ttnn.MemoryConfig | None = None,
        program_config: ttnn.MatmulProgramConfig | None = None,
        core_grid: ttnn.CoreGrid | None = None,
        output_tile: list[int] | None = None,
        dtype: ttnn.DataType | None = None,
        deallocate: bool = False,
    ) -> ttnn.Tensor:
        assert x.shape[-1] == self._in_channels, "input tensor does not have the expected shape"

        weight = self._weight
        bias = self._bias

        output = ttnn.linear(
            x,
            weight,
            bias=bias,
            memory_config=memory_config,
            program_config=program_config,
            core_grid=core_grid,
            output_tile=output_tile,
            dtype=dtype,
        )

        if deallocate:
            ttnn.deallocate(x)

        return output

    @property
    def device(self) -> ttnn.Device:
        return self._weight.device()
