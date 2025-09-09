# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Common utilities for Panoptic DeepLab.

This module provides functions used across the Panoptic DeepLab model.
"""

import torch
import ttnn


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


# Note: Legacy weight loading functions have been removed.
# Use the unified preprocess_model_parameters system in model_preprocessing.py instead.
