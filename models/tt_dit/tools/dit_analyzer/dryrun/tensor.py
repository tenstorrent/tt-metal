# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The metadata-only ``ttnn.Tensor``.

A tensor carries (logical shape, distribution, dtype, layout, IR symbol) and
nothing else. ``.shape`` reports the **per-device** shape, the way real ttnn
does, because that is what model code branches on
(``attention_ltx.py:483``: ``k_BHNE.shape[2] < _k_cos_pe.shape[2]``).

The logical shape is the analyzer's view; the local shape is the model's. Every
shape rule in :mod:`.ops` has to keep both honest -- see roadmap blocker 36.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from ..ir import Dist
from .context import CTX, TILE

#: ttnn dtype name -> analyzer dtype tag. Block-float exponent overhead is
#: phase 7 (blocker 13); the tags themselves already distinguish the formats.
DTYPES = {
    "bfloat16": "bf16",
    "float32": "fp32",
    "bfloat8_b": "bfp8_b",
    "bfloat4_b": "bfp4_b",
    "uint16": "bf16",
    "uint32": "fp32",
    "int32": "fp32",
}


def dtype_tag(dtype) -> str:
    return DTYPES.get(getattr(dtype, "name", "bfloat16"), "bf16")


class Shape(tuple):
    """Behaves like ``ttnn.Shape`` for the operations model code performs on it."""

    def __new__(cls, dims):
        return super().__new__(cls, tuple(int(d) for d in dims))

    @property
    def rank(self) -> int:
        return len(self)

    def with_tile_padding(self) -> "Shape":
        return Shape(pad_to_tile(self))

    def __eq__(self, other) -> bool:
        return tuple(self) == tuple(other) if isinstance(other, (tuple, list, Shape)) else NotImplemented

    def __ne__(self, other) -> bool:
        eq = self.__eq__(other)
        return eq if eq is NotImplemented else not eq

    def __hash__(self) -> int:
        return hash(tuple(self))


def pad_to_tile(shape: Sequence[int]) -> Tuple[int, ...]:
    """Round the last two axes up to a tile.

    A placeholder for real tiling (phase 7): enough for `get_matmul_config`'s
    arithmetic, not enough for byte accounting.
    """
    s = list(shape)
    for a in (-2, -1):
        if len(s) >= abs(a):
            s[a] = -(-s[a] // TILE) * TILE
    return tuple(s)


def local_shape(logical: Sequence[int], dist: Dist) -> Tuple[int, ...]:
    """Per-device shape: logical divided by the mesh factor on each sharded axis."""
    s = list(logical)
    for mesh_axis, tensor_axis in enumerate(dist.shard):
        if tensor_axis is None:
            continue
        a = tensor_axis % len(s)
        n = CTX.mesh.shape[mesh_axis]
        if s[a] % n:
            # Uneven shards are real (38 -> 40 heads at tp=4) and land in phase 7;
            # until the division rules match ttnn's exactly, refuse to guess.
            raise NotImplementedError(
                "uneven shard: logical %s dim %d = %d over %d devices on mesh axis %d "
                "(uneven shard division is roadmap blocker 11, phase 7)" % (list(logical), a, s[a], n, mesh_axis)
            )
        s[a] //= n
    return tuple(s)


class Tensor:
    """Metadata-only stand-in for ``ttnn.Tensor``."""

    __slots__ = ("logical", "dist", "dtype", "layout", "sym", "_host")

    def __init__(self, logical, dist: Dist, dtype=None, layout=None, sym: Optional[str] = None, host: bool = False):
        from .stubs import TILE_LAYOUT, bfloat16

        self.logical = tuple(int(d) for d in logical)
        self.dist = dist
        self.dtype = dtype or bfloat16
        self.layout = layout or TILE_LAYOUT
        self.sym = sym
        self._host = host

    # -- what the model code reads -------------------------------------------
    @property
    def shape(self) -> Shape:
        return Shape(local_shape(self.logical, self.dist))

    @property
    def logical_shape(self) -> Shape:
        """Mesh-wide shape. Not a real ttnn attribute; used by the recorder."""
        return Shape(self.logical)

    @property
    def padded_shape(self) -> Shape:
        return Shape(pad_to_tile(local_shape(self.logical, self.dist)))

    def get_dtype(self):
        return self.dtype

    def get_layout(self):
        return self.layout

    def device(self):
        return None if self._host else CTX.mesh_device

    def memory_config(self):
        from .stubs import DRAM_MEMORY_CONFIG

        return DRAM_MEMORY_CONFIG

    def is_allocated(self) -> bool:
        return True

    def spec(self):
        from .stubs import Stub

        return Stub("TensorSpec", shape=self.shape, dtype=self.dtype, layout=self.layout)

    # -- ops reachable as methods (delegated to keep one shape rule per op) ---
    def reshape(self, *shape) -> "Tensor":
        from . import ops

        if len(shape) == 1 and isinstance(shape[0], (list, tuple, Shape)):
            shape = tuple(shape[0])
        return ops.reshape(self, list(shape))

    def __add__(self, other) -> "Tensor":
        from . import ops

        return ops.pointwise("add", [self, other])

    __radd__ = __add__

    def __mul__(self, other) -> "Tensor":
        from . import ops

        return ops.pointwise("mul", [self, other])

    __rmul__ = __mul__

    def __sub__(self, other) -> "Tensor":
        from . import ops

        return ops.pointwise("sub", [self, other])

    def __truediv__(self, other) -> "Tensor":
        from . import ops

        return ops.pointwise("div", [self, other])

    def __getitem__(self, key) -> "Tensor":
        from . import ops

        if not isinstance(key, tuple):
            key = (key,)
        for axis, k in enumerate(key):
            if isinstance(k, slice) and (k.start or k.stop):
                lo = k.start or 0
                hi = k.stop if k.stop is not None else self.logical[axis]
                return ops.slice_axis(self, axis, lo, hi)
        return self

    def __repr__(self) -> str:
        return "DryRunTensor(logical=%s, local=%s, shard=%s)" % (
            list(self.logical),
            list(self.shape),
            list(self.dist.shard),
        )
