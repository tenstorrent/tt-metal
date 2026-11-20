# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Split load: a replicated cache tensor reaches the chips as 1/N slices of its own file, then is gathered.

A Parameter that no mesh axis splits (or splits only over one axis) is written in full to every chip over
that chip's own PCIe link, so a 4x8 galaxy moves 32 copies and the slowest link sets the pace. Here the
tensorbin the cache already holds is mapped, each chip is handed its 1/N byte range of it (a borrowed view of
the mapping, no conversion, no second file), the slices go up over the chips' own links, and all-gathers over
the fabric (Ethernet) rebuild the full tensor on every chip. All-gather copies bits, so the result is
bit-identical to a plain load. The gathered tensor is copied into a buffer allocated before the slices go up,
so the DRAM layout equals a plain load's (hot swap and captured traces depend on it).

A slice must be one contiguous byte range of the file: the split dim is the first one that splits evenly in
whole tiles with nothing but size-1 dims before it. Tensors that do not qualify (too small, block-float
dtypes, a split that is not contiguous) load plainly. `Module.load` and `Parameter.load` use the active
`SplitLoad` (see `utils.cache.load_model`); a module class sets `split_load = False` to opt out.
"""

from __future__ import annotations

import contextlib
import contextvars
import mmap
import os
import struct
import time
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

import ttnn

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..layers.module import Parameter

# Below this a tensor's per-chip PCIe write is noise next to a gather's fixed cost.
MIN_BYTES = 4 << 20
TILE = 32
FAILED_STAMP = "split-load-failed"

# Fixed-width dtypes only: `ttnn.host_tensor_from_bytes` does not take block-float formats.
_ITEMSIZE = {
    ttnn.bfloat16: 2,
    ttnn.float32: 4,
    ttnn.uint8: 1,
    ttnn.uint16: 2,
    ttnn.uint32: 4,
    ttnn.int32: 4,
}

_active: contextvars.ContextVar[SplitLoad | None] = contextvars.ContextVar("split_load", default=None)


def current() -> SplitLoad | None:
    return _active.get()


@contextlib.contextmanager
def disabled() -> Iterator[None]:
    """Plain loads inside this block, whatever is active outside it."""
    token = _active.set(None)
    try:
        yield
    finally:
        _active.reset(token)


def plan_for(shape, mesh_axes, mesh, dtype, layout, on_host=False) -> tuple[int, list[int]] | None:
    """How to split a tensor's replicated copies: (tensor_dim, [mesh_axis, ...]) or None.

    `shape` is the per-chip shape, `mesh_axes[d]` the mesh axis tensor dim d is already split over (or None).
    The mesh axes no tensor dim is split over hold identical copies: the tensor's first contiguous dim that
    divides over them (in whole tiles for the last two dims of a tiled tensor, never the last dim of a
    row-major one) is cut into prod(axis sizes) slices, mesh axis 0 the major index. When that does not divide,
    one axis alone is tried, the larger first. None: nothing to split, too small, or no contiguous cut.
    """
    itemsize = _ITEMSIZE.get(dtype)
    if on_host or itemsize is None or len(shape) < 2 or len(shape) > 4:
        return None
    tiled = layout == ttnn.Layout.TILE
    if tiled and (shape[-1] % TILE or shape[-2] % TILE):
        return None  # a padded tile dim: the plain load keeps the padding exactly as it was
    numel = 1
    for s in shape:
        numel *= s
    if numel * itemsize < MIN_BYTES:
        return None
    mesh = tuple(mesh)
    mapped = {a for a in mesh_axes if a is not None}
    free = [a for a, n in enumerate(mesh) if n > 1 and a not in mapped]
    candidates = [free] if len(free) > 1 else []
    candidates += [[a] for a in sorted(free, key=lambda a: -mesh[a])]
    rank = len(shape)
    for axes in candidates:
        n = 1
        for a in axes:
            n *= mesh[a]
        for d in range(rank):
            if mesh_axes[d] is not None:
                continue
            if tiled:
                before = 1
                for s in shape[:d]:
                    before *= s
                if d == rank - 1:
                    before *= shape[-2] // TILE
                part = n * TILE if d >= rank - 2 else n
            else:
                if d == rank - 1:
                    continue
                before = 1
                for s in shape[:d]:
                    before *= s
                part = n
            if before == 1 and shape[d] % part == 0:
                return d, axes
    return None


def plan(param: Parameter) -> tuple[int, list[int]] | None:
    """`plan_for` of a Parameter."""
    return plan_for(param.local_shape, param.mesh_axes, tuple(param.device.shape), param.dtype, param.layout, param.on_host)


# ---------------------------------------------------------------------------- the tensorbin header


def _u32(buf, pos: int) -> int:
    return struct.unpack_from("<I", buf, pos)[0]


def _field(buf, table: int, slot: int) -> int | None:
    """Position of a flatbuffers table's field `slot`, or None when it is absent."""
    vtable = table - struct.unpack_from("<i", buf, table)[0]
    vsize = struct.unpack_from("<H", buf, vtable)[0]
    at = 4 + 2 * slot
    if at >= vsize:
        return None
    off = struct.unpack_from("<H", buf, vtable + at)[0]
    return table + off if off else None


def read_shards(path: str | Path) -> tuple[mmap.mmap, int, dict[tuple[int, ...], tuple[int, int]]]:
    """Map a tensorbin: (mapping, start of its data region, {mesh coordinate: (offset, size) in the data region}).

    Layout (ttnn/core/tensor/serialization.cpp): a u64 header size, the flatbuffers header (a Tensor table: spec,
    mesh shape, shards, topology; a shard: buffer union type, the InlineFileStorage struct, mesh coordinate), then
    the data region. Replicated shards share one buffer, so several coordinates can name the same range.
    """
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    header_size = struct.unpack_from("<Q", mm, 0)[0]
    if 8 + header_size > len(mm):
        raise ValueError(f"'{path}' is truncated: header of {header_size} bytes in a file of {len(mm)}")
    base = 8
    root = base + _u32(mm, base)
    shards_at = _field(mm, root, 2)
    if shards_at is None:
        raise ValueError(f"'{path}' has no shards")
    vec = shards_at + _u32(mm, shards_at)
    out: dict[tuple[int, ...], tuple[int, int]] = {}
    for i in range(_u32(mm, vec)):
        elem = vec + 4 + 4 * i
        shard = elem + _u32(mm, elem)
        storage_at = _field(mm, shard, 1)
        coord_at = _field(mm, shard, 2)
        if storage_at is None or coord_at is None:
            raise ValueError(f"'{path}': shard {i} has no storage or coordinate")
        storage = storage_at + _u32(mm, storage_at)
        offset, size = struct.unpack_from("<QQ", mm, storage)
        coord = coord_at + _u32(mm, coord_at)
        values_at = _field(mm, coord, 0)
        values = values_at + _u32(mm, values_at)
        out[tuple(_u32(mm, values + 4 + 4 * j) for j in range(_u32(mm, values)))] = (offset, size)
    data_start = base + header_size
    for offset, size in out.values():
        if data_start + offset + size > len(mm):
            raise ValueError(f"'{path}' is truncated: a shard ends past the end of the file")
    return mm, data_start, out


def slice_bytes(shape, dtype, layout) -> int:
    """Bytes of one chip's tensor of `shape` (tile padding included for TILE; plan_for admits no padding)."""
    n = 1
    for s in shape:
        n *= s
    return n * _ITEMSIZE[dtype]


# ---------------------------------------------------------------------------- the loader


class SplitLoad:
    """Loads planned parameters from their cache files as 1/N slices; `ccl_manager` runs the gathers."""

    def __init__(self, ccl_manager) -> None:
        self.ccl = ccl_manager
        self.failed: str | None = None  # after a failure every later tensor loads plainly (all-or-nothing)
        self.reset_stats()

    def reset_stats(self) -> None:
        self.stats = {"split": 0, "plain": 0, "fallback": 0, "bytes": 0, "seconds": 0.0}

    @contextlib.contextmanager
    def active(self) -> Iterator[SplitLoad]:
        token = _active.set(self)
        try:
            yield self
        finally:
            _active.reset(token)

    def load(self, param: Parameter, path: Path) -> bool:
        """Fill `param` from its cache file. False: nothing was stored and the caller loads plainly."""
        splits = None if self.failed else plan(param)
        if splits is None:
            self.stats["plain"] += 1
            return False
        started = time.monotonic()
        dst = host = part = full = mm = None
        try:
            host, mm = self._host_slices(param, Path(path), *splits)
            dst = ttnn.allocate_tensor_on_device(
                ttnn.Shape(list(param.local_shape)), param.dtype, param.layout, param.device, param.memory_config
            )
            part = ttnn.to_device(host, param.device, memory_config=param.memory_config)
            ttnn.synchronize_device(param.device)  # the slices borrow the file mapping: done before it can go
            full = self._gather(part, *splits, len(param.local_shape))
            ttnn.copy(full, dst)
            for t in (full, part):
                ttnn.deallocate(t)
            full = part = None
            param.data = dst
            if os.environ.get("TT_DIT_SPLIT_LOAD_VERIFY") == "1":
                self._verify(param, dst, Path(path))
        except Exception as err:  # noqa: BLE001 — the caller (all-or-nothing per module) loads plainly
            for t in (full, part, dst):
                if t is not None:
                    with contextlib.suppress(Exception):
                        ttnn.deallocate(t)
            self.stats["fallback"] += 1
            self.failed = f"{Path(path).name}: {type(err).__name__}: {err}"
            logger.warning(f"{FAILED_STAMP}: {self.failed}; the rest of this load is plain")
            return False
        self.stats["split"] += 1
        self.stats["bytes"] += slice_bytes(param.local_shape, param.dtype, param.layout)
        self.stats["seconds"] += time.monotonic() - started
        return True

    def _verify(self, param: Parameter, dst, path: Path) -> None:
        """TT_DIT_SPLIT_LOAD_VERIFY=1: every chip's copy must equal a plain load of the same file, byte for byte.
        A mismatch raises, so the module falls back to the plain load and stamps `split-load-failed`."""
        import torch

        ref = ttnn.load_tensor(str(path), device=param.device)
        try:
            for i, (a, b) in enumerate(zip(ttnn.get_device_tensors(dst), ttnn.get_device_tensors(ref))):
                if not torch.equal(ttnn.to_torch(a).view(torch.uint8), ttnn.to_torch(b).view(torch.uint8)):
                    raise ValueError(f"chip {i} differs from a plain load")
        finally:
            ttnn.deallocate(ref)

    def _host_slices(self, param: Parameter, path: Path, dim: int, axes: list[int]):
        """The host tensor holding, per chip, its slice of the file: borrowed views of the file's mapping."""
        import numpy as np

        mm, data_start, shards = read_shards(path)
        mesh = tuple(param.device.shape)
        n = 1
        for a in axes:
            n *= mesh[a]
        full_bytes = slice_bytes(param.local_shape, param.dtype, param.layout)
        piece_shape = list(param.local_shape)
        piece_shape[dim] //= n
        piece = slice_bytes(piece_shape, param.dtype, param.layout)
        if piece * n != full_bytes:
            raise ValueError(f"slice bytes {piece} x {n} != tensor bytes {full_bytes}")
        spec = ttnn.TensorSpec(ttnn.Shape(piece_shape), param.dtype, param.layout, param.memory_config)
        views: dict[tuple[int, int], object] = {}
        rows, cols = mesh
        for r in range(rows):
            for c in range(cols):
                offset, size = shards[(r, c)]
                if size != full_bytes:
                    raise ValueError(f"file shard {(r, c)} holds {size} bytes, the tensor needs {full_bytes}")
                k = 0
                for a in axes:
                    k = k * mesh[a] + (r, c)[a]
                key = (offset, k)
                if key not in views:
                    arr = np.frombuffer(mm, dtype=np.uint8, count=piece, offset=data_start + offset + k * piece)
                    views[key] = ttnn.host_tensor_from_bytes(arr, spec)
                views[(r, c)] = views[key]
        shard_list = [views[(r, c)] for r in range(rows) for c in range(cols)]
        return ttnn.from_host_shards(shard_list, ttnn.MeshShape(rows, cols)), mm

    def _gather(self, part, dim, axes, rank):
        """All-gather `part` over the split axes (highest axis first); each gather copies bits."""
        out = part
        try:
            for axis in sorted(axes, reverse=True):
                nxt = self.ccl.all_gather(out, dim=dim - rank, mesh_axis=axis, use_hyperparams=False)
                if out is not part:
                    ttnn.deallocate(out)
                out = nxt
        except Exception:
            if out is not part:
                with contextlib.suppress(Exception):
                    ttnn.deallocate(out)
            raise
        return out
