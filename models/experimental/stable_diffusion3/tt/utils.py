# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch
import ttnn
from loguru import logger


def allocate_tensor_on_device_like(
    t: ttnn.Tensor, *, device: ttnn.Device, memory_config: ttnn.MemoryConfig | None = None
) -> ttnn.Tensor:
    return ttnn.allocate_tensor_on_device(t.shape, t.dtype, t.layout, device, memory_config=memory_config)


def to_torch(tensor, mesh_device, dtype, shard_dim=-1):
    return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=shard_dim), dtype=dtype)


def from_torch_fast(
    t: torch.Tensor,
    *,
    device: ttnn.Device | ttnn.MeshDevice | None = None,
    layout: ttnn.Layout | None = None,
    dtype: ttnn.DataType | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
    to_host: bool = False,
    mesh_mapper: ttnn.TensorToMesh | None = None,
    shard_dim: int | None = None,
) -> ttnn.Tensor:
    if shard_dim is not None:
        mesh_mapper = ttnn.ShardTensorToMesh(device, dim=shard_dim)

    if isinstance(device, ttnn.MeshDevice) and mesh_mapper is None:
        mesh_mapper = ttnn.ReplicateTensorToMesh(device)

    # ttnn.to_layout does not support changing the datatype or memory_config if the layout already matches. ttnn.clone
    # does not support changing the datatype if the input is not tiled. An option could be to tilize the input before
    # changing the datatype and then untilize again, but it was not tested if this would be faster than converting the
    # datatype on the host. Also ttnn.to_dtype does not support device tensors.
    if device is None or layout is None or layout == ttnn.ROW_MAJOR_LAYOUT:
        return ttnn.from_torch(
            t,
            device=None if to_host else device,
            layout=layout,
            dtype=dtype,
            memory_config=memory_config,
            mesh_mapper=mesh_mapper,
        )

    tensor = ttnn.from_torch(t, device=device, mesh_mapper=mesh_mapper)

    if tensor.shape[-2] == 32 and t.shape[-2] == 1:
        # Work around the fact that the shape is erroneously set to the padded shape under certain conditions.
        assert isinstance(device, ttnn.MeshDevice)
        assert dtype in (ttnn.bfloat4_b, ttnn.bfloat8_b)
        tensor = tensor.reshape(ttnn.Shape(t.shape))

    tensor = ttnn.to_layout(tensor, layout, dtype=dtype, memory_config=memory_config)

    if to_host:
        tensor = tensor.cpu()

    return tensor


def to_memory_config(
    t: ttnn.Tensor, memory_config: ttnn.MemoryConfig, *, dtype: ttnn.DataType | None = None, deallocate: bool = False
) -> ttnn.Tensor:
    result = ttnn.to_memory_config(t, memory_config, dtype=dtype)

    result_is_same = result.memory_config() == t.memory_config() and result.buffer_address() == t.buffer_address()
    if deallocate and not result_is_same:
        ttnn.deallocate(t)

    return result


def assert_quality(
    a: ttnn.Tensor | torch.Tensor,
    b: ttnn.Tensor | torch.Tensor,
    *,
    num_devices: int | None = None,
    pcc: float | None = None,
    mse: float | None = None,
    shard_dim: int | None = None,
) -> None:
    if isinstance(a, ttnn.Tensor):
        a = to_torch(a, mesh_device=a.device(), dtype=a.get_dtype(), shard_dim=shard_dim)
        if num_devices is not None:
            assert shard_dim == 0
            a = a[0 : a.shape[0] // num_devices, ...]
    if isinstance(b, ttnn.Tensor):
        b = to_torch(b, mesh_device=b.device(), dtype=b.get_dtype(), shard_dim=shard_dim)
        if num_devices is not None:
            assert shard_dim == 0
            b = b[0 : b.shape[0] // num_devices, ...]

    if math.prod(a.shape) != math.prod(b.shape):
        msg = f"incompatible shapes: {a.shape} != {b.shape}"
        raise ValueError(msg)

    if a.shape != b.shape:
        logger.warning(f"shape mismatch: {a.shape} != {b.shape}")

    a = a.detach().flatten().to(torch.float32)
    b = b.detach().flatten().to(torch.float32)

    cov = torch.cov(torch.stack([a, b])).numpy()

    std_a = math.sqrt(cov[0, 0])
    std_b = math.sqrt(cov[1, 1])
    pcc_calculated = cov[0, 1] / (std_a * std_b)
    beta = cov[0, 1] / cov[0, 0]
    mean_a = a.mean().item()
    mean_b = b.mean().item()

    mse_calculated = torch.nn.functional.mse_loss(a, b).item()

    logger.info(f"μ₁ = {mean_a:.3g}, μ₂ = {mean_b:.3g}, σ₁ = {std_a:.3g}, σ₂ = {std_b:.3g}")
    logger.info(f"PCC = {pcc_calculated * 100:.4f} %, MSE = {mse_calculated:.3g}, β = {beta * 100:.0f} %")
    if pcc is not None:
        assert pcc_calculated >= pcc, f"PCC = {pcc_calculated * 100:.4f} % >= {pcc * 100:.4f} %"
    if mse is not None:
        assert mse_calculated <= mse, f"MSE = {mse_calculated:.3g} <= {mse:.3g}"


def all_gather(
    x: ttnn.Tensor,
    dim: int,
    *,
    cluster_axis: int | None = None,
    mesh_device: ttnn.MeshDevice | None = None,
    num_links: int = 1,
    topology: ttnn.Topology = ttnn.Topology.Ring,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    assert cluster_axis is None or mesh_device is not None, "cluster_axis requires mesh_device to be set"
    assert x.shape[dim] == x.padded_shape[dim], f"dimension {dim} of {x.shape} should not be padded"

    shape = list(x.shape)

    reshape = x.shape != x.padded_shape
    if reshape:
        x = ttnn.reshape(x, x.padded_shape, x.padded_shape)

    if cluster_axis is not None:
        x = ttnn.all_gather(
            x,
            dim,
            cluster_axis,
            mesh_device,
            num_links=num_links,
            topology=topology,
            memory_config=memory_config,
        )

    x = ttnn.all_gather(
        x,
        dim,
        num_links=num_links,
        topology=topology,
        memory_config=memory_config,
    )

    if reshape:
        shape[dim] = x.shape[dim]
        x = ttnn.reshape(x, shape, x.padded_shape)

    return x
