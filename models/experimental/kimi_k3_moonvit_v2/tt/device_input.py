# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device inputs with a fixed address, refilled from the host for each image.

A captured trace replays the device addresses it recorded, so every per-image input (patches,
position embeddings, RoPE tables, `logical_n`) must live in one device tensor that is written in
place, never reallocated. `DeviceInput` allocates on its first write and copies into the same
tensor on every later one (`ttnn.copy_host_to_device_tensor`).

The host side is a staging buffer allocated once, in the device's byte layout (tiled for TILE), that a
host tensor borrows (`ttnn.host_tensor_from_bytes`, one shard per chip, `ttnn.from_local_host_shards`).
A write fills the staging buffer in place and copies it to the chips: no host tensor is built and nothing
is converted or tilized by ttnn per image. `write_rows` and `fill_rows` fill only the rows they are given
and zero the rows a previous write left behind, so the rows after them are zero as with `write`, at the
cost of the rows that changed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch

import ttnn

# Element type of the staging buffer per device dtype. uint32 is held as int32: the same bits for the values used.
_STORAGE_DTYPES = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.uint32: torch.int32}
TILE = 32
FACE = 16


def sp_mesh_mapper(mesh_device: ttnn.MeshDevice, sp_axis: int | None):
    """Splits dim 2 (the sequence) along mesh axis `sp_axis`, or replicates when `sp_axis` is None."""
    if sp_axis is None:
        return ttnn.ReplicateTensorToMesh(mesh_device)
    dims = [None, None]
    dims[sp_axis] = 2
    return ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)


def tiles_view(flat: torch.Tensor, width: int) -> torch.Tensor:
    """A flat buffer in ttnn's TILE layout -> view [tile rows, tile cols, 2, 2, 16, 16]: 32x32 tiles in row-major
    order, each four 16x16 faces in row-major order, each face row major."""
    return flat.view(-1, width // TILE, 2, 2, FACE, FACE)


def tilize_into(dst: torch.Tensor, rows: torch.Tensor) -> None:
    """Write `rows` [m, width], m a multiple of 32, into the tiles view `dst` (`tiles_view`) of m / 32 tile rows,
    converting to dst's dtype."""
    m, width = rows.shape
    dst.copy_(rows.view(m // TILE, 2, FACE, width // TILE, 2, FACE).permute(0, 3, 1, 4, 2, 5))


def untilize(flat: torch.Tensor, width: int) -> torch.Tensor:
    """The inverse of `tilize_into` for a whole buffer -> [rows, width]."""
    tiles = tiles_view(flat, width)
    return tiles.permute(0, 2, 4, 1, 3, 5).reshape(-1, width)


class DeviceInput:
    """One fixed-shape [1, 1, rows, width] input, split along `sp_axis` on dim 2 or replicated when `sp_axis` is None.

    TILE inputs need rows per chip and width to be multiples of 32."""

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
        if len(self.shape) != 4 or self.shape[:2] != (1, 1):
            raise ValueError(f"expected a [1, 1, rows, width] input, got {self.shape}")
        if dtype not in _STORAGE_DTYPES:
            raise ValueError(f"unsupported device input dtype {dtype}")
        self.rows, self.width = self.shape[2:]
        self.mesh_device = mesh_device
        self.sp_axis = sp_axis
        self.dtype = dtype
        self.layout = layout
        self.tiled = layout == ttnn.TILE_LAYOUT
        if self.tiled and (self.rows % TILE or self.width % TILE):
            raise ValueError(f"a TILE input needs whole tiles, got {self.shape}")
        self.tensor: ttnn.Tensor | None = None
        self._staging: torch.Tensor | None = None  # flat, in the device's byte layout
        self._host: ttnn.Tensor | None = None  # borrows _staging
        # Rows of the staging buffer, from the top, that may hold non-zero values (whole tile rows for TILE).
        self._dirty = 0

    @property
    def storage_dtype(self) -> torch.dtype:
        return _STORAGE_DTYPES[self.dtype]

    def _staging_buffer(self) -> torch.Tensor:
        if self._staging is None:
            self._staging = torch.zeros(self.rows * self.width, dtype=self.storage_dtype)
        return self._staging

    def host_value(self) -> torch.Tensor:
        """The staging buffer as the logical [1, 1, rows, width] tensor: what the last write sent to the chips."""
        flat = self._staging_buffer()
        rows = untilize(flat, self.width) if self.tiled else flat.view(self.rows, self.width)
        return rows.reshape(self.shape)

    def write(self, value: torch.Tensor) -> ttnn.Tensor:
        """Fill the tensor with `value` (any shape with `prod(shape)` elements) and return it.

        A float `value` is cast to the device dtype in torch: ttnn's own fp32 -> bf16 conversion is much slower."""
        return self.write_rows(value.reshape(self.rows, self.width))

    def write_rows(self, rows: torch.Tensor) -> ttnn.Tensor:
        """Fill the first `len(rows)` rows with `rows` [n, width] and zero the rest, then copy to the chips and return
        the device tensor. Only the rows a previous write left non-zero are cleared."""
        n = rows.shape[0]
        if rows.ndim != 2 or rows.shape[1] != self.width or n > self.rows:
            raise ValueError(f"expected at most {self.rows} rows of width {self.width}, got {tuple(rows.shape)}")
        flat = self._staging_buffer()
        self._mark_dirty(n)
        if not self.tiled:
            flat.view(self.rows, self.width)[:n].copy_(rows)
            return self._finish(n)
        whole = n - n % TILE
        tiles = tiles_view(flat, self.width)
        if whole:
            tilize_into(tiles[: whole // TILE], rows[:whole])
        if whole < n:
            last = torch.zeros(TILE, self.width, dtype=rows.dtype)
            last[: n - whole] = rows[whole:]
            tilize_into(tiles[whole // TILE : whole // TILE + 1], last)
        return self._finish(n)

    def fill_rows(self, n: int, fill: Callable[[torch.Tensor], None]) -> ttnn.Tensor:
        """ROW_MAJOR only: `fill(view)` writes the first `n` rows in place through `view` [n, width] of the staging
        buffer; the rest is zeroed as in `write_rows`. Saves the copy of a separate [n, width] tensor."""
        if self.tiled:
            raise ValueError("fill_rows writes row-major inputs; use write_rows for TILE")
        if not 0 <= n <= self.rows:
            raise ValueError(f"{n} rows for an input of {self.rows}")
        staging = self._staging_buffer()
        self._mark_dirty(n)
        fill(staging.view(self.rows, self.width)[:n])
        return self._finish(n)

    def _whole_rows(self, n: int) -> int:
        """Staging rows that n logical rows occupy: whole tile rows for TILE."""
        return -(-n // TILE) * TILE if self.tiled else n

    def _mark_dirty(self, n: int) -> None:
        """Before writing n rows: if the write stops part way, the next one still clears them."""
        self._dirty = max(self._dirty, self._whole_rows(n))

    def _finish(self, n: int) -> ttnn.Tensor:
        """Zero the stale rows after the first n, then upload."""
        n = self._whole_rows(n)
        if n < self._dirty:
            self._staging.view(self.rows, self.width)[n : self._dirty].zero_()
        self._dirty = n
        return self._upload()

    def _upload(self) -> ttnn.Tensor:
        if self.tensor is None:
            # Allocated once, from the logical value, so the tensor's topology is ttnn's own.
            self.tensor = ttnn.from_torch(
                self.host_value(),
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=self.dtype,
                layout=self.layout,
                mesh_mapper=sp_mesh_mapper(self.mesh_device, self.sp_axis),
            )
        else:
            if self._host is None:
                self._host = self._borrowing_host_tensor()
            ttnn.copy_host_to_device_tensor(self._host, self.tensor)
        return self.tensor

    def _borrowing_host_tensor(self) -> ttnn.Tensor:
        """A host tensor over the staging buffer, with no copy: one shard per chip under the device tensor's topology,
        or, replicated, one tensor that the copy writes to every chip."""
        raw = self._staging.view(torch.uint8).numpy()
        if self.sp_axis is None:
            spec = ttnn.TensorSpec(ttnn.Shape(self.shape), self.dtype, self.layout, ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.host_tensor_from_bytes(raw, spec)
        mesh_shape = tuple(self.mesh_device.shape)
        sp = mesh_shape[self.sp_axis]
        shard_rows = self.rows // sp
        spec = ttnn.TensorSpec(
            ttnn.Shape((1, 1, shard_rows, self.width)), self.dtype, self.layout, ttnn.DRAM_MEMORY_CONFIG
        )
        # Rows are contiguous in both layouts (TILE: whole tile rows), so chip k's shard is the k-th slice.
        shard_bytes = raw.size // sp
        shards = {}
        for coord in (tuple(c) for c in _mesh_coords(mesh_shape)):
            k = coord[self.sp_axis]
            shards[ttnn.MeshCoordinate(*coord)] = ttnn.host_tensor_from_bytes(
                raw[k * shard_bytes : (k + 1) * shard_bytes], spec
            )
        return ttnn.from_local_host_shards(shards, self.mesh_device, self.tensor.tensor_topology())


def _mesh_coords(mesh_shape: Sequence[int]):
    """Every coordinate of a mesh, row major."""
    return [divmod(i, mesh_shape[1]) for i in range(math.prod(mesh_shape))]
