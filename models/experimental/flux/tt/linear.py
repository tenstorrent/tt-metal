# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from . import utils
from .utils import from_torch_fast


@dataclass
class LinearParameters:
    """A container for the parameters of a linear layer."""

    weight: ttnn.Tensor
    bias: ttnn.Tensor | None
    on_host: bool
    device: ttnn.MeshDevice
    reduce_scatter: bool

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
        on_host: bool = False,
        unsqueeze_bias: bool = False,
        mesh_sharding_dim: int | None = None,
        chunks: int | None = None,
    ) -> LinearParameters:
        """Creates a LinearParameters instance from a torch state dictionary.

        This method converts torch tensors to ttnn tensors and adapts them for use as linear layer
        parameters.

        `mesh_sharding_dim` can have three possible values: A value of 0 means that the weight
        matrix will be sharded over its input dimension. In this case, the input tensor is required
        to be sharded along its last dimension. The output tensor will be sharded over its last
        dimension as well. A value of 1 means that the weight matrix will be sharded along its
        output dimension. This value requires the input tensor to be replicated over the mesh. The
        output tensor will be sharded over its last dimension. A value of None means that no
        sharding of the weight matrix is performed.

        Sharding of the weight matrix and output tensor is performed in the second dimension of the
        mesh grid. Sharding of the input tensor is expected to be the same. In addition, input
        tensors can be sharded in the first dimension of the mesh grid, which will be preserved by
        the linear operation.
        """
        _, mesh_width = device.shape

        weight = state["weight"]
        assert len(weight.shape) == 2, "weight should be a rank two tensor"

        if "bias" in state:
            bias = state["bias"]
            assert len(bias.shape) == 1, "bias should be a rank one tensor"
        else:
            bias = None

        if chunks is not None:
            _, in_dim = weight.shape

            n = mesh_width
            weight = weight.reshape([chunks, n, -1, in_dim]).permute([1, 0, 2, 3]).reshape([-1, in_dim])
            bias = bias.reshape([chunks, n, -1]).permute([1, 0, 2]).reshape([-1]) if bias is not None else None

        if unsqueeze_bias:
            # TODO: Remove this workaround for issue https://github.com/tenstorrent/tt-metal/issues/16599
            bias = bias.unsqueeze(0)

        on_host = on_host or device is None

        if mesh_sharding_dim is None:
            weight_mm = bias_mm = ttnn.ReplicateTensorToMesh(device)
            output_sharding = False
        elif mesh_sharding_dim == 1:
            weight_mm = bias_mm = ttnn.ShardTensor2dMesh(device, tuple(device.shape), (None, -1))
            output_sharding = False
        elif mesh_sharding_dim == 0:
            weight_mm = ttnn.ShardTensor2dMesh(device, tuple(device.shape), (None, -2))
            if bias is not None:
                mesh_height, mesh_width = device.shape
                zeros = torch.zeros_like(bias)
                bias = torch.cat([bias] + [zeros] * (mesh_width - 1), dim=-1)
                bias_mm = ttnn.ShardTensor2dMesh(device, mesh_shape=(mesh_height, mesh_width), dims=(None, -1))
            else:
                bias_mm = None
            output_sharding = True
        else:
            msg = "mesh_sharding_dim must be in the range from -2 to 1, or None"
            raise ValueError(msg)

        return cls(
            weight=from_torch_fast(
                weight.transpose(0, 1),
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
                to_host=on_host,
                mesh_mapper=weight_mm,
            ),
            bias=from_torch_fast(
                bias.unsqueeze(0),
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
                to_host=on_host,
                mesh_mapper=bias_mm,
            )
            if bias is not None
            else None,
            on_host=on_host,
            device=device,
            reduce_scatter=output_sharding and mesh_width > 1,
        )

    @property
    def in_channels(self) -> int:
        return self.weight.shape[0]

    @property
    def out_channels(self) -> int:
        return self.weight.shape[1]


class Linear:
    def __init__(self, parameters: LinearParameters) -> None:
        self._reduce_scatter = parameters.reduce_scatter
        self._in_channels = parameters.in_channels
        self._weight = parameters.weight
        self._bias = parameters.bias
        self._paramters_on_host = parameters.on_host
        self._device = parameters.device

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        memory_config: ttnn.MemoryConfig | None = None,
        program_config: ttnn.MatmulProgramConfig | None = None,
        core_grid: ttnn.CoreGrid | None = None,
        output_tile: list[int] | None = None,
        dtype: ttnn.DataType | None = None,
        activation: str | None = None,
        skip_reduce_scatter: bool = False,
    ) -> ttnn.Tensor:
        msg = f"last value in input shape {list(x.shape)} should be equal to {self._in_channels}"
        assert x.shape[-1] == self._in_channels, msg

        if self._reduce_scatter:
            msg = "activation function can not be fused when reduce_scatter is performed afterwards"
            assert activation == None, msg

        if memory_config is None:
            memory_config = x.memory_config()

        if self._paramters_on_host:
            weight = self._weight.to(self._device)
            bias = self._bias.to(self._device) if self._bias is not None else None
        else:
            weight = self._weight
            bias = self._bias

        x = ttnn.linear(
            x,
            weight,
            bias=bias,
            memory_config=memory_config,
            program_config=program_config,
            core_grid=core_grid,
            output_tile=output_tile,
            dtype=dtype,
            activation=activation,
        )

        if skip_reduce_scatter:
            return x

        return self.reduce_scatter(x, memory_config=memory_config)

    def reduce_scatter(
        self,
        x: ttnn.Tensor,
        memory_config: ttnn.MemoryConfig | None = None,
        scatter_dim: int = -1,
    ) -> ttnn.Tensor:
        if not self._reduce_scatter:
            return x

        return utils.reduce_scatter(
            x,
            dim=scatter_dim,
            math_op=ttnn.ReduceType.Sum,
            cluster_axis=1,
            mesh_device=self._device,
            # reduce_scatter currently requires linear topology when specifying a cluster axis
            topology=ttnn.Topology.Linear,
            memory_config=memory_config,
        )


class _ShardBias:
    """A mesh mapper for sharding the bias of a linear operation.

    This mesh mapper is intended for sharding the bias of a linear operation on the first dimension.
    A single device receive the bias as is, while the other ones receive zero tensors of the same
    shape so that the bias is not added multiple times after gathering.

    The otherwise problematic behavior of adding the bias mutiple times is currently not observed
    with a bias of type bfloat8_b or bfloat4_b, since ttnn.from_torch pads to such tensors to the
    tile size before sharding, which has the same effect if the number devices is not too big.
    """

    def __init__(self, mesh_device: ttnn.MeshDevice) -> None:
        self.mesh_device = mesh_device

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
