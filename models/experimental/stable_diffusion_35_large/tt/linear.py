# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass

# if TYPE_CHECKING:
import torch
import ttnn

from .utils import from_torch_fast


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
        shard_dim: int = None,
        unsqueeze_bias: bool = False,
    ) -> TtLinearParameters:
        if "bias" in state:
            bias = state["bias"].unsqueeze(0)
        else:
            bias = None
        weight = state["weight"]
        if os.environ["MESH_DEVICE"] == "T3K":
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
            if weight_w % hidden_dim == 0:
                if weight_w_mult == 1:
                    weight = torch.nn.functional.pad(weight, pad=(0, hidden_dim_pad), mode="constant", value=0)
                elif weight_w_mult > 1:
                    weight = weight.reshape(weight_h, weight_w_mult, hidden_dim)
                    weight = torch.nn.functional.pad(weight, pad=(0, hidden_dim_pad), mode="constant", value=0)
                    weight = weight.reshape(weight_h, weight_w_mult * hidden_dim_new)

            if not bias == None:
                bias_h, bias_w = bias.shape
                bias_w_mult = bias_w // hidden_dim
                if bias_w % hidden_dim == 0:
                    if bias_w_mult == 1:
                        bias = torch.nn.functional.pad(bias, pad=(0, hidden_dim_pad), mode="constant", value=0)
                    elif bias_w_mult > 1:
                        bias = bias.reshape(bias_h, bias_w_mult, hidden_dim)
                        bias = torch.nn.functional.pad(bias, pad=(0, hidden_dim_pad), mode="constant", value=0)
                        bias = bias.reshape(bias_h, bias_w_mult * hidden_dim_new)

        if unsqueeze_bias:
            # TODO: Once the issue is resolved, remove this workaround for https://github.com/tenstorrent/tt-metal/issues/16599
            bias = bias.unsqueeze(0)

        if shard_dim in [0, -2]:
            bias_mm = _ShardBias(device)
        elif shard_dim in [1, -1]:
            bias_mm = ttnn.ShardTensorToMesh(device, 1)
        else:
            bias_mm = ttnn.ReplicateTensorToMesh(device)

        return cls(
            weight=from_torch_fast(
                weight.transpose(0, 1),
                dtype=dtype,
                device=device,
                shard_dim=shard_dim,
                layout=ttnn.TILE_LAYOUT,
            ),
            bias=from_torch_fast(
                bias,
                dtype=dtype,
                device=device,
                mesh_mapper=bias_mm,
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
        unpadded_num_heads: int,
        hidden_dim_padding: int,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtLinearParameters:
        if "bias" in state:
            torch_bias = state["bias"].unsqueeze(0)
        else:
            torch_bias = None

        weight = state["weight"]
        torch_weight = weight.transpose(0, 1)

        if os.environ["MESH_DEVICE"] == "T3K":
            head_size = torch_weight.shape[1] // 3 // unpadded_num_heads
            head_padding = device.get_num_devices() - (unpadded_num_heads % device.get_num_devices())
            weight_h, weight_w = torch_weight.shape

            torch_weight = torch_weight.reshape(weight_h, 3, unpadded_num_heads, head_size)
            torch_weight = torch.nn.functional.pad(
                torch_weight, pad=(0, 0, 0, head_padding, 0, 0, 0, hidden_dim_padding), mode="constant", value=0
            )
            torch_weight = torch_weight.reshape(weight_h + hidden_dim_padding, -1)
            if not torch_bias == None:
                bias_h, bias_w = torch_bias.shape
                torch_bias = torch_bias.reshape(bias_h, 3, unpadded_num_heads, head_size)
                torch_bias = torch.nn.functional.pad(torch_bias, pad=(0, 0, 0, head_padding), mode="constant", value=0)
                torch_bias = torch_bias.reshape(bias_h, -1)

        def shuffle_heads(tensor):
            # Given torch tensor with output features in the last dimension,
            # shuffle heads to allow for column parallel computation
            in_dim = tensor.shape[0]
            tensor = tensor.reshape(in_dim, 3, device.get_num_devices(), n_local_heads, -1)  # [ID, 3, ND, NLH, DH]
            tensor = tensor.permute(0, 2, 1, 3, 4)  # [ID, ND, 3, NLH, DH]
            tensor = tensor.reshape(in_dim, -1)  # [ID, ND*3*NLH*DH]
            return tensor

        return cls(
            weight=from_torch_fast(
                shuffle_heads(torch_weight),
                dtype=dtype,
                device=device,
                shard_dim=-1,
                layout=ttnn.TILE_LAYOUT,
            ),
            bias=from_torch_fast(
                shuffle_heads(torch_bias),
                dtype=dtype,
                device=device,
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
        hidden_dim_padding: int,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
        unsqueeze_bias: bool = False,
    ) -> TtLinearParameters:
        if "bias" in state:
            torch_bias = state["bias"].unsqueeze(0)
        else:
            torch_bias = None
        weight = state["weight"]
        torch_weight = weight.transpose(0, 1)

        if os.environ["MESH_DEVICE"] == "T3K":
            weight_h, weight_w = torch_weight.shape
            torch_weight = torch_weight.reshape(weight_h, num_chunks, -1)
            torch_weight = torch.nn.functional.pad(
                torch_weight, pad=(0, hidden_dim_padding, 0, 0, 0, hidden_dim_padding), mode="constant", value=0
            )
            torch_weight = torch_weight.reshape(weight_h + hidden_dim_padding, -1)

            if not torch_bias == None:
                bias_h, bias_w = torch_bias.shape
                torch_bias = torch_bias.reshape(bias_h, num_chunks, -1)
                torch_bias = torch.nn.functional.pad(torch_bias, pad=(0, hidden_dim_padding), mode="constant", value=0)
                torch_bias = torch_bias.reshape(bias_h, -1)

        def shuffle_chunks(tensor):
            # Given torch tensor with output features in the last dimension,
            # shuffle heads to allow for column parallel computation
            in_dim = tensor.shape[0]
            tensor = tensor.reshape(in_dim, num_chunks, device.get_num_devices(), -1)
            tensor = tensor.permute(0, 2, 1, 3)
            tensor = tensor.reshape(in_dim, -1)
            return tensor

        torch_weight = shuffle_chunks(torch_weight)
        torch_bias = shuffle_chunks(torch_bias)

        if unsqueeze_bias:
            # TODO: Once the issue is resolved, remove this workaround for https://github.com/tenstorrent/tt-metal/issues/16599
            torch_bias = torch_bias.unsqueeze(0)

        return cls(
            weight=from_torch_fast(
                torch_weight,
                dtype=dtype,
                device=device,
                shard_dim=-1,
                layout=ttnn.TILE_LAYOUT,
            ),
            bias=from_torch_fast(
                torch_bias,
                dtype=dtype,
                device=device,
                shard_dim=-1,
                layout=ttnn.TILE_LAYOUT,
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
        activation: str | None = None,
        deallocate: bool = False,
        highest_quality: bool = True,
    ) -> ttnn.Tensor:
        msg = f"last value in input shape {list(x.shape)} should be equal to {self._in_channels}"
        assert x.shape[-1] == self._in_channels, msg

        weight = self._weight
        bias = self._bias

        # there is a correctness issue with tensors of shape Mx1x1xN, squeeze them to Mx1xN
        squeeze = len(x.shape) == 4 and x.shape[1] == 1 and x.shape[2] == 1
        if squeeze:
            x = x.reshape([x.shape[0], 1, x.shape[-1]])

        assert x.device() == weight.device()
        if bias is not None:
            assert x.device() == bias.device()

        output = ttnn.linear(
            x,
            weight,
            bias=bias,
            memory_config=memory_config,
            program_config=program_config,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            if highest_quality
            else None,
            core_grid=core_grid,
            output_tile=output_tile,
            dtype=dtype,
            activation=activation,
        )

        if deallocate:
            ttnn.deallocate(x)

        if squeeze:
            output = output.reshape([output.shape[0], 1, 1, output.shape[-1]])

        return output


class _ShardBias(ttnn.TensorToMesh):
    """A mesh mapper for sharding the bias of a linear operation.

    This mesh mapper is intended for sharding the bias of a linear operation on the first dimension.
    A single device receive the bias as is, while the other ones receive zero tensors of the same
    shape so that the bias is not added multiple times after gathering.
    """

    def __init__(self, mesh_device: ttnn.MeshDevice) -> None:
        super().__init__(mesh_device)

    def map(self, tensor: torch.Tensor) -> dict[int, ttnn.Tensor]:
        mesh_height, mesh_width = self.mesh_device.shape

        zeros = torch.zeros_like(tensor)
        return ([tensor] + [zeros] * (mesh_width - 1)) * mesh_height

    def config(self) -> dict[str, str]:
        mesh_height, mesh_width = self.mesh_device.shape

        return {
            "strategy": "shard_2d",
            "mesh_shape_y": str(mesh_height),
            "mesh_shape_x": str(mesh_width),
        }
