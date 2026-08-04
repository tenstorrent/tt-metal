# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Throwaway metadata-only ``torch`` for the dry-run spike.

Only exists because this machine has no torch. A real dry run should use real
torch with ``device='meta'`` tensors instead -- same idea, no fake to maintain.
Covers just the surface the tt_dit import path and module construction touch.
"""

from __future__ import annotations

import types
from typing import Any, List, Sequence, Tuple


class _DType:
    def __init__(self, name: str, itemsize: int = 2):
        self.name = name
        self.itemsize = itemsize

    def __repr__(self) -> str:
        return "torch." + self.name


bfloat16 = _DType("bfloat16", 2)
float16 = _DType("float16", 2)
float32 = _DType("float32", 4)
float64 = _DType("float64", 8)
int32 = _DType("int32", 4)
int64 = _DType("int64", 8)
long = int64
bool_ = _DType("bool", 1)
uint8 = _DType("uint8", 1)
uint16 = _DType("uint16", 2)
uint32 = _DType("uint32", 4)
int8 = _DType("int8", 1)
int16 = _DType("int16", 2)
globals()["bool"] = bool_  # torch.bool


class Tensor:
    """Shape + dtype only. Every op returns another metadata tensor."""

    def __init__(self, shape: Sequence[int], dtype: _DType = bfloat16):
        self._shape = tuple(int(d) for d in shape)
        self.dtype = dtype

    # -- shape ---------------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def size(self, dim: int = None):
        return self._shape if dim is None else self._shape[dim]

    def numel(self) -> int:
        n = 1
        for d in self._shape:
            n *= d
        return n

    def dim(self) -> int:
        return len(self._shape)

    # -- shape-preserving / shape-changing ops -------------------------------
    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = tuple(shape[0])
        shape = list(shape)
        if -1 in shape:
            known = 1
            for d in shape:
                if d != -1:
                    known *= d
            shape[shape.index(-1)] = self.numel() // max(1, known)
        return Tensor(shape, self.dtype)

    view = reshape

    def permute(self, *dims) -> "Tensor":
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = tuple(dims[0])
        return Tensor([self._shape[d] for d in dims], self.dtype)

    def transpose(self, a: int, b: int) -> "Tensor":
        s = list(self._shape)
        s[a], s[b] = s[b], s[a]
        return Tensor(s, self.dtype)

    @property
    def T(self) -> "Tensor":
        return Tensor(tuple(reversed(self._shape)), self.dtype)

    def unsqueeze(self, dim: int) -> "Tensor":
        s = list(self._shape)
        s.insert(dim if dim >= 0 else len(s) + dim + 1, 1)
        return Tensor(s, self.dtype)

    def squeeze(self, dim: int = None) -> "Tensor":
        if dim is None:
            return Tensor([d for d in self._shape if d != 1], self.dtype)
        s = list(self._shape)
        if s[dim] == 1:
            s.pop(dim)
        return Tensor(s, self.dtype)

    def index_select(self, dim: int, index: "Tensor") -> "Tensor":
        s = list(self._shape)
        s[dim] = index.numel() if isinstance(index, Tensor) else len(index)
        return Tensor(s, self.dtype)

    def chunk(self, n: int, dim: int = 0) -> List["Tensor"]:
        s = list(self._shape)
        s[dim] = s[dim] // n
        return [Tensor(s, self.dtype) for _ in range(n)]

    def to(self, *a, **k) -> "Tensor":
        dtype = k.get("dtype") or next((x for x in a if isinstance(x, _DType)), self.dtype)
        return Tensor(self._shape, dtype)

    def float(self) -> "Tensor":
        return Tensor(self._shape, float32)

    def bfloat16(self) -> "Tensor":
        return Tensor(self._shape, bfloat16)

    def contiguous(self) -> "Tensor":
        return self

    def clone(self) -> "Tensor":
        return Tensor(self._shape, self.dtype)

    def detach(self) -> "Tensor":
        return self

    def cpu(self) -> "Tensor":
        return self

    # -- arithmetic (metadata only) ------------------------------------------
    def _binary(self, other) -> "Tensor":
        if isinstance(other, Tensor) and other.numel() > self.numel():
            return Tensor(other._shape, self.dtype)
        return Tensor(self._shape, self.dtype)

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = _binary

    def sum(self, dim: int = None, **k) -> "Tensor":
        if dim is None:
            return Tensor([], self.dtype)
        s = list(self._shape)
        s.pop(dim)
        return Tensor(s, self.dtype)

    def max(self, *a, **k) -> "Tensor":
        return Tensor([], self.dtype)

    def int(self) -> "Tensor":
        return Tensor(self._shape, int32)

    def item(self) -> int:
        # A dry run has no values. Anything that shapes the graph from a value
        # must be supplied by the caller; see roadmap blocker 39.
        raise NotImplementedError("torch.Tensor.item() in a dry run: value-dependent shape")

    # -- indexing ------------------------------------------------------------
    def __getitem__(self, key) -> "Tensor":
        if not isinstance(key, tuple):
            key = (key,)
        out = []
        for axis, k in enumerate(key):
            if isinstance(k, slice):
                start, stop, _ = k.indices(self._shape[axis])
                out.append(stop - start)
            elif k is Ellipsis:
                out.extend(self._shape[axis:])
        out.extend(self._shape[len(key) :])
        return Tensor(out or [1], self.dtype)

    def __setitem__(self, key, value) -> None:
        pass  # metadata only: writes have no effect

    def __repr__(self) -> str:
        return "FakeTorchTensor(%s, %s)" % (list(self._shape), self.dtype)


# -- factories ---------------------------------------------------------------
def _factory(*shape, **k) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    return Tensor(shape, k.get("dtype") or bfloat16)


zeros = ones = empty = _factory


def full(shape, value, **k) -> Tensor:
    return Tensor(shape, k.get("dtype") or bfloat16)


def arange(*a, **k) -> Tensor:
    start, stop, step = 0, a[0], 1
    if len(a) >= 2:
        start, stop = a[0], a[1]
    if len(a) >= 3:
        step = a[2]
    return Tensor([max(0, (stop - start) // step)], k.get("dtype") or int64)


def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    s = list(tensors[0].shape)
    s[dim] = sum(t.shape[dim] for t in tensors)
    return Tensor(s, tensors[0].dtype)


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    s = list(tensors[0].shape)
    s.insert(dim, len(tensors))
    return Tensor(s, tensors[0].dtype)


def tensor(data, **k) -> Tensor:
    def shape_of(x):
        return [len(x)] + shape_of(x[0]) if isinstance(x, (list, tuple)) and x else []

    return Tensor(shape_of(data), k.get("dtype") or float32)


def _noop(*a, **k) -> Any:
    return None


def install() -> types.ModuleType:
    """Register the fake as ``torch`` (plus the submodules tt_dit imports)."""
    import sys

    mod = types.ModuleType("torch")
    for name, value in globals().items():
        if not name.startswith("_") or name in ("_DType",):
            setattr(mod, name, value)
    mod.bool = bool_
    mod.Tensor = Tensor
    mod.device = lambda *a, **k: "meta"
    mod.no_grad = lambda: _NullCtx()
    mod.inference_mode = lambda *a, **k: _NullCtx()
    mod.manual_seed = _noop
    mod.set_grad_enabled = _noop
    nn = types.ModuleType("torch.nn")
    nn.functional = types.ModuleType("torch.nn.functional")
    nn.functional.pad = lambda x, *a, **k: x
    nn.Module = object
    mod.nn = nn
    sys.modules["torch"] = mod
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = nn.functional
    return mod


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
