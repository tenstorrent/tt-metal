# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from typing import Any

import torch
import tracy
import ttnn
from loguru import logger

ENABLE_SIGNPOSTS = os.environ.get("ENABLE_SIGNPOSTS") == "1"


def allocate_tensor_on_device_like(
    t: ttnn.Tensor, *, device: ttnn.Device, memory_config: ttnn.MemoryConfig | None = None
) -> ttnn.Tensor:
    return ttnn.allocate_tensor_on_device(t.shape, t.dtype, t.layout, device, memory_config=memory_config)


def from_torch_fast(
    t: torch.Tensor,
    *,
    device: ttnn.Device | ttnn.MeshDevice | None = None,
    layout: ttnn.Layout | None = None,
    dtype: ttnn.DataType | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
    to_host: bool = False,
    mesh_mapper: ttnn.TensorToMesh | None = None,
) -> ttnn.Tensor:
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


def assert_quality(
    a: ttnn.Tensor | torch.Tensor,
    b: ttnn.Tensor | torch.Tensor,
    *,
    pcc: float | None = None,
    mse: float | None = None,
    mesh_composer: ttnn.MeshToTensor | None = None,
) -> None:
    if isinstance(a, ttnn.Tensor):
        a = ttnn.to_torch(a, mesh_composer=mesh_composer)
    if isinstance(b, ttnn.Tensor):
        b = ttnn.to_torch(b, mesh_composer=mesh_composer)

    assert a.shape == b.shape, f"{a.shape} != {b.shape}"

    a = a.detach().to(torch.float32)
    b = b.detach().to(torch.float32)

    cov = torch.cov(torch.stack([a.flatten(), b.flatten()])).numpy()

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

    if cluster_axis is not None:
        return ttnn.all_gather(
            x,
            dim,
            cluster_axis,
            mesh_device,
            num_links=num_links,
            topology=topology,
            memory_config=memory_config,
        )

    return ttnn.all_gather(
        x,
        dim,
        num_links=num_links,
        topology=topology,
        memory_config=memory_config,
    )


def reduce_scatter(
    x: ttnn.Tensor,
    dim: int,
    math_op: ttnn.ReduceType,
    *,
    cluster_axis: int | None = None,
    mesh_device: ttnn.MeshDevice | None = None,
    num_links: int = 1,
    topology: ttnn.Topology = ttnn.Topology.Ring,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    assert cluster_axis is None or mesh_device is not None, "cluster_axis requires mesh_device to be set"

    if memory_config is None:
        memory_config = x.memory_config()

    # ttnn.reduce_scatter currently supports tensors of rank 4 only
    rank = len(x.shape)
    if rank < 4:
        shape = [1] * (4 - rank) + list(x.shape)
        x = ttnn.reshape(x, shape)
        if dim >= 0:
            dim += 4 - rank

    if dim not in {3, -1}:  # https://github.com/tenstorrent/tt-metal/issues/19433
        x = ttnn.transpose(x, dim, 3)

    if cluster_axis is not None:
        x = ttnn.reduce_scatter(
            x,
            3,
            cluster_axis,
            mesh_device,
            math_op,
            num_links=num_links,
            topology=topology,
            memory_config=memory_config,
        )
    else:
        x = ttnn.reduce_scatter(
            x,
            3,
            math_op,
            num_links=num_links,
            topology=topology,
            memory_config=memory_config,
        )

    if dim not in {3, -1}:  # https://github.com/tenstorrent/tt-metal/issues/19433
        x = ttnn.transpose(x, dim, 3)

    if rank < 4:
        shape = list(x.shape)[4 - rank :]
        x = ttnn.reshape(x, shape)

    return x


def signpost(header: str, message: Any = None) -> None:  # noqa: ANN401
    if ENABLE_SIGNPOSTS:
        tracy.signpost(header, message)
