# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device inputs with a fixed address, refilled from the host for each image.

A captured trace replays the device addresses it recorded, so every per-image input (patches,
position embeddings, RoPE tables, `logical_n`) must live in one device tensor that is written in
place, never reallocated. `DeviceInput` allocates on its first `write` and copies into the same
tensor on every later one (`ttnn.copy_host_to_device_tensor`).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import ttnn


_TORCH_FLOAT_DTYPES = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}


class DeviceInput:
    """One fixed-shape input, split along `sp_axis` on dim 2 or replicated when `sp_axis` is None."""

    def __init__(
        self,
        shape: Sequence[int],
        *,
        mesh_device: ttnn.MeshDevice,
        sp_axis: int | None,
        dtype: ttnn.DataType = ttnn.bfloat16,
        layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    ) -> None:
        self.shape = tuple(shape)
        self.mesh_device = mesh_device
        self.sp_axis = sp_axis
        self.dtype = dtype
        self.layout = layout
        self.tensor: ttnn.Tensor | None = None

    def _mesh_mapper(self):
        if self.sp_axis is None:
            return ttnn.ReplicateTensorToMesh(self.mesh_device)
        dims = [None, None]
        dims[self.sp_axis] = 2
        return ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=dims)

    def write(self, value: torch.Tensor) -> ttnn.Tensor:
        """Fill the tensor with `value` (any shape with `prod(shape)` elements) and return it.

        A float `value` is cast to the device dtype in torch first: ttnn's own fp32 -> bf16 conversion
        is much slower (full-size patches: ~350 ms against ~11 ms for the cast)."""
        value = value.reshape(self.shape)
        torch_dtype = _TORCH_FLOAT_DTYPES.get(self.dtype)
        if torch_dtype is not None and value.is_floating_point():
            value = value.to(torch_dtype)
        kwargs = dict(dtype=self.dtype, layout=self.layout, mesh_mapper=self._mesh_mapper())
        if self.tensor is None:
            self.tensor = ttnn.from_torch(
                value, device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kwargs
            )
        else:
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(value, **kwargs), self.tensor)
        return self.tensor
