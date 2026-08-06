# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""A metadata-only ``torch``, used only when the interpreter has no real torch.

A dry run wants `torch.empty(shape, device='meta')`: shapes without bytes, which
real torch does better than any fake. This module exists so the analyzer still
runs on a bare interpreter (no torch, no numpy) -- CI for the dry run is meant to
be a unit-test-cost job on a machine with no Tenstorrent hardware, and requiring
a torch install would work against that.

It covers only the slice of torch that tt_dit's import path and module
construction touch. :func:`dit_analyzer.dryrun.hostenv.ensure_host_env` prefers
real torch whenever it can import it.
"""

from __future__ import annotations

import sys
import types
from typing import Any, List, Sequence, Tuple


class DType:
    def __init__(self, name: str, itemsize: int = 2):
        self.name = name
        self.itemsize = itemsize

    def __repr__(self) -> str:
        return "torch." + self.name


bfloat16 = DType("bfloat16", 2)
float16 = DType("float16", 2)
float32 = DType("float32", 4)
float64 = DType("float64", 8)
int8 = DType("int8", 1)
int16 = DType("int16", 2)
int32 = DType("int32", 4)
int64 = DType("int64", 8)
long = int64
uint8 = DType("uint8", 1)
uint16 = DType("uint16", 2)
uint32 = DType("uint32", 4)
bool_ = DType("bool", 1)


class Tensor:
    """Shape + dtype only. Every op returns another metadata tensor."""

    def __init__(self, shape: Sequence[int], dtype: DType = bfloat16):
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

    def flip(self, *a, **k) -> "Tensor":
        return self

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

    def repeat(self, *reps) -> "Tensor":
        if len(reps) == 1 and isinstance(reps[0], (list, tuple)):
            reps = tuple(reps[0])
        s = [1] * (len(reps) - len(self._shape)) + list(self._shape)
        return Tensor([d * r for d, r in zip(s, reps)], self.dtype)

    def to(self, *a, **k) -> "Tensor":
        dtype = k.get("dtype") or next((x for x in a if isinstance(x, DType)), self.dtype)
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

    # elementwise ops broadcast to the larger operand's shape; `theta ** arange(...)`
    # (rope inverse-frequencies) needs the reflected forms too.
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _binary
    __truediv__ = __rtruediv__ = __pow__ = __rpow__ = __floordiv__ = __rfloordiv__ = __mod__ = _binary

    def __neg__(self) -> "Tensor":
        return Tensor(self._shape, self.dtype)

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
        raise NotImplementedError("torch.Tensor.item() in a dry run: value-dependent shape (roadmap blocker 39)")

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
        return "MetaTorchTensor(%s, %s)" % (list(self._shape), self.dtype)


# -- factories ---------------------------------------------------------------
def _factory(*shape, **k) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    return Tensor(shape, k.get("dtype") or bfloat16)


zeros = ones = empty = _factory


def full(shape, value, **k) -> Tensor:
    return Tensor(shape, k.get("dtype") or bfloat16)


def arange(*a, **k) -> Tensor:
    # torch.arange accepts positional (stop) / (start, stop) / (start, stop, step)
    # *and* keyword start=/end=/step= (the H3 time-embedding uses arange(start=0, end=half_dim)).
    start, stop, step = 0, None, 1
    if len(a) == 1:
        stop = a[0]
    elif len(a) >= 2:
        start, stop = a[0], a[1]
        if len(a) >= 3:
            step = a[2]
    start = k.get("start", start)
    stop = k.get("end", k.get("stop", stop))
    step = k.get("step", step)
    if stop is None:
        raise TypeError("arange: missing stop/end")
    n = -(-(stop - start) // step) if step > 0 else 0  # ceil division, matching torch's length
    return Tensor([max(0, n)], k.get("dtype") or int64)


def _unary(x, *a, **k) -> Tensor:
    """Shape-preserving elementwise op (torch.exp/log/sin/... on a host constant)."""
    return Tensor(list(x.shape), x.dtype) if isinstance(x, Tensor) else Tensor([])


# host-constant elementwise ops used while building time / rope embeddings (Timesteps,
# rope factors). All preserve shape; only the shape matters to the dry run.
exp = exp2 = log = log2 = log10 = sin = cos = tan = sqrt = rsqrt = sigmoid = tanh = erf = reciprocal = _unary


def outer(a: Tensor, b: Tensor, *r, **k) -> Tensor:
    return Tensor([a.numel(), b.numel()], a.dtype)


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


class _NullCtx:
    """A no-op usable both as a context manager and as a decorator.

    ``torch.no_grad()`` / ``inference_mode()`` are used both ways in tt_dit
    (``with torch.no_grad():`` and ``@torch.no_grad()`` on a method, e.g.
    ``VAEDecoderAdapter.decode``); the decorator form calls the returned object,
    so it must pass the wrapped function through unchanged.
    """

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __call__(self, fn=None, *a, **k):
        return fn if callable(fn) else self


def install() -> types.ModuleType:
    """Register this module as ``torch`` (plus the submodules tt_dit imports)."""
    mod = types.ModuleType("torch")
    for name, value in globals().items():
        if not name.startswith("_"):
            setattr(mod, name, value)
    mod.bool = bool_
    mod.Tensor = Tensor
    mod.dtype = DType
    mod.device = lambda *a, **k: "meta"
    mod.no_grad = lambda: _NullCtx()
    mod.inference_mode = lambda *a, **k: _NullCtx()
    mod.manual_seed = _noop
    mod.set_grad_enabled = _noop
    mod.pow = _unary  # torch.pow(x, y) — shape of x; not the builtin pow
    mod.clamp = mod.clip = _unary
    mod.maximum = mod.minimum = lambda a, b, *r, **k: _unary(a)
    nn = types.ModuleType("torch.nn")
    nn.functional = types.ModuleType("torch.nn.functional")
    nn.functional.pad = lambda x, *a, **k: x
    nn.Module = object
    mod.nn = nn
    sys.modules["torch"] = mod
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = nn.functional
    return mod
