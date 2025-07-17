# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch
import ttnn
from loguru import logger


def create_global_semaphores(mesh_device, num_devices, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = ttnn.create_global_semaphore(mesh_device, cores, initial_value)
    return ccl_semaphore_handles


def allocate_tensor_on_device_like(
    t: ttnn.Tensor, *, device: ttnn.Device, memory_config: ttnn.MemoryConfig | None = None
) -> ttnn.Tensor:
    return ttnn.allocate_tensor_on_device(t.shape, t.dtype, t.layout, device, memory_config=memory_config)


def to_torch(
    tensor: torch.Tensor,
    *,
    mesh_device: ttnn.MeshDevice | None = None,  # this is only used to construct a mesh composer
    dtype: torch.dtype | None = None,
    mesh_composer: ttnn.CppMeshToTensor | None = None,
    shard_dim: int | None = None,
    fix_special_numbers: bool = False,
) -> torch.Tensor:
    if shard_dim is not None:
        mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=shard_dim)

    if mesh_composer is None:
        tensor = ttnn.get_device_tensors(tensor)[0]

    result = ttnn.to_torch(tensor, mesh_composer=mesh_composer)

    # ttnn.to_torch ignores the dtype argument if a mesh composer is supplied
    if dtype is not None:
        result = result.to(dtype)

    if not fix_special_numbers:
        return result

    assert dtype in [torch.float32, torch.float64]

    mask = torch.isnan(result) | torch.isinf(result)
    negative = torch.signbit(result)

    finfo = torch.finfo(dtype)
    result[mask.logical_and(~negative)] = finfo.max
    result[mask.logical_and(negative)] = finfo.min

    return result


def from_torch_fast(
    t: torch.Tensor,
    *,
    device: ttnn.Device | ttnn.MeshDevice | None = None,
    layout: ttnn.Layout | None = None,
    dtype: ttnn.DataType | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
    to_host: bool = False,
    mesh_mapper: ttnn.TensorToMesh | None = None,
    # The argument shard_dim is a bit problematic. If set, it creates a mesh mapper with the given
    # device. But for a host tensor, the device is None, so a mesh mapper can not be created.
    shard_dim: int | None = None,
) -> ttnn.Tensor:
    assert shard_dim is None or device is not None, "shard_dim requires device"

    if isinstance(device, ttnn.MeshDevice):
        if shard_dim is not None:
            mesh_mapper = ttnn.ShardTensorToMesh(device, dim=shard_dim)
        if mesh_mapper is None:
            mesh_mapper = ttnn.ReplicateTensorToMesh(device)
    elif isinstance(device, ttnn.Device):
        mesh_mapper = None

    float32_in = t.dtype == torch.float32
    float32_out = dtype == ttnn.float32 or (dtype is None and float32_in)

    # ttnn.to_layout does not support changing the datatype or memory_config if the layout already matches. ttnn.clone
    # does not support changing the datatype if the input is not tiled. An option could be to tilize the input before
    # changing the datatype and then untilize again, but it was not tested if this would be faster than converting the
    # datatype on the host. Also ttnn.to_dtype does not support device tensors. Additionally, `ttnn.to_layout` is lossy
    # for float32.
    if device is None or layout is None or layout == ttnn.ROW_MAJOR_LAYOUT or (float32_in and float32_out):
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


def from_torch_fast_2d(
    t: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    mesh_shape: Tuple[int, int],
    dims: Tuple[Optional[int], Optional[int]],
    *,
    layout: ttnn.Layout | None = None,
    dtype: ttnn.DataType | None = None,
    mesh_mapper: ttnn.TensorToMesh | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    if mesh_mapper is None and (dims[0] is not None or dims[1] is not None):
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=dims)

    return ttnn.from_torch(
        t,
        device=mesh_device,
        layout=layout,
        dtype=dtype,
        memory_config=memory_config,
        mesh_mapper=mesh_mapper,
    )


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
    ccc: float | None = None,
    mse: float | None = None,
    relative_rmse: float | None = None,
    shard_dim: int | None = None,
) -> None:
    if isinstance(a, ttnn.Tensor):
        a = to_torch(
            a,
            mesh_device=a.device(),
            dtype=torch.float32,
            shard_dim=shard_dim,
            fix_special_numbers=True,
        )
        if num_devices is not None:
            assert shard_dim == 0
            a = a[0 : a.shape[0] // num_devices, ...]

    if isinstance(b, ttnn.Tensor):
        b = to_torch(
            b,
            mesh_device=b.device(),
            dtype=torch.float32,
            shard_dim=shard_dim,
            fix_special_numbers=True,
        )
        if num_devices is not None:
            assert shard_dim == 0
            b = b[0 : b.shape[0] // num_devices, ...]

    if math.prod(a.shape) != math.prod(b.shape):
        msg = f"incompatible shapes: {a.shape} != {b.shape}"
        raise ValueError(msg)

    if a.shape != b.shape:
        logger.warning(f"shape mismatch: {a.shape} != {b.shape}")

    a = a.detach().flatten().to(torch.float64)
    b = b.detach().flatten().to(torch.float64)

    cov = torch.cov(torch.stack([a, b])).numpy()

    std_a = math.sqrt(cov[0, 0])
    std_b = math.sqrt(cov[1, 1])
    mean_a = a.mean().item()
    mean_b = b.mean().item()

    pcc_found = cov[0, 1] / (std_a * std_b)
    beta_found = cov[0, 1] / cov[0, 0]
    ccc_found = 2 * pcc_found * std_a * std_b / (std_a**2 + std_b**2 + (mean_a - mean_b) ** 2)
    relative_rmse_found = torch.nn.functional.mse_loss(a, b).sqrt().item() / std_a

    if mse is not None:
        relative_rmse = math.sqrt(mse) / std_a

    logger.info(f"μ₁ = {mean_a:.3g}, μ₂ = {mean_b:.3g}, σ₁ = {std_a:.3g}, σ₂ = {std_b:.3g}")
    logger.info(
        f"PCC = {pcc_found * 100:.4f} %, "
        f"β = {beta_found * 100:.1f} %, "
        f"CCC = {ccc_found * 100:.4f} %, "
        f"RMSE/σ₁ = {relative_rmse_found * 100:.1f} %"
    )

    if pcc is not None and pcc_found < pcc:
        msg = f"PCC = {pcc_found * 100:.4f} % >= {pcc * 100:.4f} %"
        raise Exception(msg)  # noqa: TRY002

    if ccc is not None and ccc_found < ccc:
        msg = f"CCC = {ccc_found * 100:.4f} % >= {ccc * 100:.4f} %"
        raise Exception(msg)  # noqa: TRY002

    if relative_rmse is not None and relative_rmse_found > relative_rmse:
        msg = f"RMSE/σ₁ = {relative_rmse_found * 100:.1f} % <= {relative_rmse * 100:.1f} %"
        raise Exception(msg)  # noqa: TRY002


def all_gather(
    x: ttnn.Tensor,
    dim: int,
    topology: ttnn.Topology,
    *,
    cluster_axis: int | None = None,
    mesh_device: ttnn.MeshDevice | None = None,
    num_links: int = 1,
    memory_config: ttnn.MemoryConfig | None = None,
    multi_device_global_semaphore,
    parallel_config: DiTParallelConfig,
) -> ttnn.Tensor:
    assert cluster_axis is None or mesh_device is not None, "cluster_axis requires mesh_device to be set"
    assert x.shape[dim] == x.padded_shape[dim], f"dimension {dim} of {x.shape} should not be padded"

    shape = list(x.shape)

    reshape = x.shape != x.padded_shape
    if reshape:
        x = ttnn.reshape(x, x.padded_shape, x.padded_shape)

    if cluster_axis is not None:
        x = ttnn.experimental.all_gather_async(
            x,
            dim=dim,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            topology=topology,
            multi_device_global_semaphore=multi_device_global_semaphore,
            memory_config=memory_config,
            num_links=num_links,
        )

    x = ttnn.experimental.all_gather_async(
        x,
        dim,
        cluster_axis=parallel_config.tensor_parallel.mesh_axis,
        mesh_device=mesh_device,
        topology=topology,
        multi_device_global_semaphore=multi_device_global_semaphore,
        memory_config=memory_config,
        num_links=num_links,
    )

    if reshape:
        shape[dim] = x.shape[dim]
        x = ttnn.reshape(x, shape, x.padded_shape)

    return x


def silu(x: ttnn.Tensor) -> ttnn.Tensor:
    """More accurate version of `ttnn.silu`"""
    return ttnn.div(x, ttnn.exp(ttnn.neg(x)) + 1)


def unpadded_all_gather_async(
    x,
    dim,
    cluster_axis,
    mesh_device,
    topology,
    multi_device_global_semaphore,
    memory_config=None,
    num_links=1,
    persistent_output_tensor=None,
):
    shape = list(x.shape)

    x = ttnn.experimental.all_gather_async(
        x,
        dim=dim,
        cluster_axis=cluster_axis,
        mesh_device=mesh_device,
        topology=topology,
        multi_device_global_semaphore=multi_device_global_semaphore,
        memory_config=memory_config,
        num_links=num_links,
        persistent_output_tensor=persistent_output_tensor,
    )

    shape[dim] = x.shape[dim]
    x = ttnn.reshape(x, shape, x.padded_shape)

    return x
