# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from . import utils
from .utils import from_torch_fast


@dataclass
class TtRmsNormParameters:
    weight: ttnn.Tensor

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtRmsNormParameters:
        return cls(
            weight=from_torch_fast(state["weight"].unsqueeze(0), layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
        )


class TtRmsNorm:
    def __init__(self, parameters: TtRmsNormParameters, *, eps: float) -> None:
        super().__init__()

        self._eps = eps
        self._weight = parameters.weight

    def __call__(self, x: ttnn.Tensor, *, deallocate: bool = False) -> ttnn.Tensor:
        output = ttnn.rms_norm(
            x,
            weight=self._weight,
            epsilon=self._eps,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        )

        if deallocate:
            ttnn.deallocate(x)

        return output


@dataclass
class TtLayerNormParameters:
    device: ttnn.MeshDevice
    weight: ttnn.Tensor | None = None
    bias: ttnn.Tensor | None = None
    distributed: bool = False

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.MeshDevice,
        distributed: bool = True,
        weight_shape: list[int] | None = None,
    ) -> TtLayerNormParameters:
        _, mesh_width = device.shape
        distributed = distributed and mesh_width > 1

        weight = state.get("weight")
        bias = state.get("bias")

        if distributed:
            # ttnn.layer_norm_post_all_gather currently requires weight and bias
            if weight is None:
                assert weight_shape is not None, "weight_shape is required when weight is missing"
                weight = torch.ones(weight_shape)
            if bias is None:
                bias = torch.zeros_like(weight)

            h = 32 * mesh_width
            _, mesh_width = device.shape
            weight = weight.reshape([-1, h])
            bias = bias.reshape([-1, h])

        mesh_mapper = ttnn.ShardTensor2dMesh(device, tuple(device.shape), (None, -1)) if distributed else None
        layout = ttnn.ROW_MAJOR_LAYOUT if distributed else ttnn.TILE_LAYOUT
        if distributed and dtype != ttnn.float32:
            dtype = ttnn.bfloat16

        return cls(
            weight=from_torch_fast(
                weight,
                layout=layout,
                dtype=dtype,
                device=device,
                mesh_mapper=mesh_mapper,
            )
            if weight is not None
            else None,
            bias=from_torch_fast(
                bias,
                layout=layout,
                dtype=dtype,
                device=device,
                mesh_mapper=mesh_mapper,
            )
            if bias is not None
            else None,
            distributed=distributed,
            device=device,
        )


class TtLayerNorm:
    def __init__(self, parameters: TtLayerNormParameters, *, eps: float) -> None:
        super().__init__()

        self._eps = eps
        self._distributed = parameters.distributed
        self._weight = parameters.weight
        self._bias = parameters.bias
        self._device = parameters.device

    def __call__(
        self,
        x: ttnn.Tensor,
        memory_config: ttnn.MemoryConfig | None = None,
        program_config: ttnn.ProgramConfig | None = None,
    ) -> ttnn.Tensor:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        if not self._distributed:
            return ttnn.layer_norm(
                x,
                weight=self._weight,
                bias=self._bias,
                epsilon=self._eps,
                memory_config=memory_config,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
            )

        rank = len(x.shape)
        if rank < 4:
            shape = [1] * (4 - rank) + list(x.shape)
            x = ttnn.reshape(x, shape)

        stats = ttnn.layer_norm_pre_all_gather(
            x,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
        )

        stats = utils.all_gather(
            stats,
            dim=-1,
            mesh_device=self._device,
            # all_gather currently requires linear topology when specifying a cluster axis
            topology=ttnn.Topology.Linear,
        )

        x = ttnn.layer_norm_post_all_gather(
            x,
            stats,
            weight=self._weight,
            bias=self._bias,
            epsilon=self._eps,
            memory_config=memory_config,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

        if rank < 4:
            shape = list(x.shape)[4 - rank :]
            x = ttnn.reshape(x, shape)

        return x
