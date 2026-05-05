# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


def from_torch_host_to_device(
    tensor: torch.Tensor,
    *,
    device,
    dtype=None,
    layout=None,
    memory_config=None,
    mesh_mapper=None,
    **kwargs,
):
    host_kwargs = dict(kwargs)
    if dtype is not None:
        host_kwargs["dtype"] = dtype
    if layout is not None:
        host_kwargs["layout"] = layout
    if mesh_mapper is not None:
        host_kwargs["mesh_mapper"] = mesh_mapper

    host_tensor = ttnn.from_torch(tensor, device=None, **host_kwargs)
    if device is None:
        return host_tensor

    to_device_kwargs = {"device": device}
    if memory_config is not None:
        to_device_kwargs["memory_config"] = memory_config
    return ttnn.to_device(host_tensor, **to_device_kwargs)


def prepare_residual_tensor_prefill_host_to_device(x_bsh: torch.Tensor, configuration, force_replicated=False):
    dims = (None, None) if force_replicated else (None, -1)
    mesh_mapper = ttnn.ShardTensor2dMesh(
        configuration.mesh_device,
        dims=dims,
        mesh_shape=configuration.cluster_shape,
    )

    return from_torch_host_to_device(
        x_bsh.unsqueeze(0),
        device=configuration.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
