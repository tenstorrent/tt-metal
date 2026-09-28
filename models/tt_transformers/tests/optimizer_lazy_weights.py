# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Deferred checkpoint tensors: a stored weight skips the model's torch work, not only its conversion.

The weight store (optimizer_weight_cache) keys a converted weight by the bytes of the torch tensor the
model hands to `ttnn.as_tensor`. To have those bytes, the model's torch work had to run first: every
transpose copy, concatenation and pad, on every run, for a weight that was already on disk. That work
grew with the model code the optimizer writes: Qwen3-32B's merged, zero-padded gate/up weight costs
0.48 s of host copies per layer, 30.6 of the 32.5 s a warm run spent outside the store (2026-09-28).

Here every checkpoint tensor the model reads is a `LazyWeight`: a tensor with the checkpoint tensor's
shape, dtype and strides but no data, that records each torch op applied to it instead of running it
(the op, its non-tensor arguments, and its tensor arguments, themselves recorded). The result of a
recorded op is another LazyWeight, its shape and strides computed by running the op on meta tensors.
So when the model calls `ttnn.as_tensor(w13)`, the store holds the recipe for w13, not its bytes:

    recipe(w13) = pad(cat([reshape(transpose(ckpt[w1])), reshape(transpose(ckpt[w3]))]), (0, 768))

A recipe determines its bytes exactly: the ops are deterministic CPU ops and replaying them on the
same inputs is what the model would have run. Its key hashes the op names, every argument, and the
content of each checkpoint tensor it starts from. The store remembers, per recipe key, the content
key of the weight the recipe produced, and a later run with the same recipe loads that file without
running the recipe. A recipe the store has not seen is replayed ("materialized") into an ordinary
tensor and converted exactly as before, and its content key recorded.

What cannot be deferred runs eagerly, with the same result the model would have had:
  - an op without a meta kernel, a random op, one with an argument this module cannot describe
    exactly, or one with a large ordinary tensor argument: its LazyWeight inputs are materialized
    and the op runs on real tensors;
  - reading a value (`.item()`, `.tolist()`, a comparison): materialized, then read.
Two things would make deferral differ from eager execution, and both stop it ("taint") instead:
an op that writes into a checkpoint tensor or a tensor derived from one (a deferred recipe replayed
after the write would see the new values), and a real tensor that shares memory with one (a view
returned by an eager op, `.numpy()`, `.data_ptr()`). A tainted build raises LazyUnsupported, and the
harness (RunCache.build) rebuilds the model with laziness off -- the exact eager path.
"""

from __future__ import annotations

import json
from collections import Counter

import torch
from torch.utils._pytree import tree_flatten, tree_map

# Ordinary (non-lazy) tensor arguments up to this size are copied into the recipe; a larger one runs
# its op eagerly, since the copy would cost what deferral saves.
CONST_MAX_BYTES = 64 << 20


class LazyUnsupported(RuntimeError):
    """The model did something deferral cannot reproduce exactly; the build is redone eagerly."""


class _State:
    def __init__(self):
        self.taint: str | None = None
        self.sources = 0
        self.recorded = 0
        self.eager: Counter = Counter()  # "op: reason" -> count, for the report


STATE = _State()


def reset() -> None:
    global STATE
    STATE = _State()


def _taint(reason: str) -> LazyUnsupported:
    if STATE.taint is None:
        STATE.taint = reason
    return LazyUnsupported(f"deferred checkpoint tensor: {reason}")


# ---- hashing tensor content ----------------------------------------------------------------------
def memory_span(data):
    """The memory a tensor reads, as a flat view, and its strides from the start of that memory.

    The bytes from the tensor's first element to its last, with its shape and strides, determine its
    contents exactly, so a key can hash that span in place instead of copying the tensor.
    """
    if data.numel() == 0:
        return None, list(data.stride())
    if any(s < 0 for s in data.stride()):  # not produced by torch's own views; copy to be safe
        data = data.contiguous()
    extent = 1 + sum((n - 1) * s for n, s in zip(data.shape, data.stride()))
    span = torch.as_strided(data, (extent,), (1,))  # starts at the tensor's first element
    return span, list(data.stride())


def content_digest(tensor, tag: str = "") -> str:
    import xxhash

    data = tensor.detach()
    span, strides = memory_span(data)
    h = xxhash.xxh3_128()
    h.update(json.dumps([tag, str(data.dtype), list(data.shape), strides]).encode())
    if span is not None and span.numel():
        h.update(span.view(torch.uint8).numpy())
    return h.hexdigest()


# ---- describing arguments exactly ----------------------------------------------------------------
class _Opaque(Exception):
    pass


def _describe(value):
    """A JSON value that identifies a non-tensor argument exactly, or _Opaque."""
    if value is None:
        return None
    if isinstance(value, bool):
        return ["b", value]
    if isinstance(value, int):
        return ["i", value]
    if isinstance(value, float):
        return ["f", value.hex()]
    if isinstance(value, str):
        return ["s", value]
    if isinstance(value, (torch.dtype, torch.device, torch.memory_format, torch.layout)):
        return ["t", repr(value)]
    if isinstance(value, (list, tuple, torch.Size)):
        return [type(value).__name__, [_describe(v) for v in value]]
    raise _Opaque(type(value).__name__)


class _Const:
    """An ordinary tensor argument of a recorded op, copied so later writes to it cannot reach the recipe."""

    def __init__(self, tensor):
        self.tensor = tensor.detach().clone()
        self.digest = content_digest(self.tensor, "const")


class _Source:
    def __init__(self, tensor):
        self.tensor = tensor
        self._digest = None

    def digest(self) -> str:
        if self._digest is None:
            self._digest = content_digest(self.tensor, "source")
        return self._digest


class _Call:
    def __init__(self, func, args, kwargs):
        self.func, self.args, self.kwargs = func, args, kwargs
        self._digest = None

    def digest(self) -> str:
        if self._digest is None:
            import xxhash

            def leaf(x):
                if isinstance(x, LazyWeight):
                    return ["lazy", x.recipe_digest()]
                if isinstance(x, _Const):
                    return ["const", x.digest]
                return _describe(x)

            doc = [str(self.func), tree_map(leaf, list(self.args)), tree_map(leaf, dict(sorted(self.kwargs.items())))]
            self._digest = xxhash.xxh3_128(json.dumps(doc).encode()).hexdigest()
        return self._digest


# ---- the tensor ----------------------------------------------------------------------------------
def _meta(x):
    if isinstance(x, torch.Tensor):
        return torch.empty_strided(tuple(x.shape), tuple(x.stride()), dtype=x.dtype, device="meta")
    if isinstance(x, _Const):
        return _meta(x.tensor)
    return x


def _is_write(arg) -> bool:
    return arg.alias_info is not None and arg.alias_info.is_write


def _args_by_schema(func, args, kwargs):
    """(schema argument, value) for every argument the call passed."""
    schema = func._schema.arguments
    pairs = [(schema[i], v) for i, v in enumerate(args) if i < len(schema)]
    by_name = {a.name: a for a in schema}
    pairs += [(by_name[k], v) for k, v in kwargs.items() if k in by_name]
    return pairs


def _has_lazy(value) -> bool:
    return any(isinstance(x, LazyWeight) for x in tree_flatten(value)[0])


def _is_tensor_type(t) -> bool:
    """A schema return of Tensor or List[Tensor] (never Tensor?, a scalar or a tuple of mixed types)."""
    if isinstance(t, torch._C.TensorType):
        return True
    return isinstance(t, torch._C.ListType) and isinstance(t.getElementType(), torch._C.TensorType)


def _why_not_recorded(func, args, kwargs) -> str | None:
    if torch.Tag.nondeterministic_seeded in func.tags:
        return "random"
    if any(_is_write(a) for a in func._schema.arguments):
        return "writes an argument"
    if not func._schema.returns or not all(_is_tensor_type(r.type) for r in func._schema.returns):
        return "returns a non-tensor"
    for x in tree_flatten((args, kwargs))[0]:
        if isinstance(x, LazyWeight):
            continue
        if isinstance(x, torch.Tensor):
            if x.device.type != "cpu" or x.requires_grad or x.untyped_storage().nbytes() > CONST_MAX_BYTES:
                return "large or non-cpu tensor argument"
            continue
        try:
            _describe(x)
        except _Opaque as exc:
            return f"argument of type {exc}"
    return None


def materialize(value, memo: dict | None = None):
    """`value` with every LazyWeight in it replayed into an ordinary tensor."""
    memo = {} if memo is None else memo

    def real(x):
        if isinstance(x, LazyWeight):
            node = x._node
            if isinstance(node, _Source):
                return node.tensor
            if id(node) not in memo:
                try:
                    memo[id(node)] = node.func(*tree_map(real, list(node.args)), **tree_map(real, dict(node.kwargs)))
                except Exception as exc:
                    raise _taint(f"replaying {node.func} failed ({type(exc).__name__}: {exc})") from exc
            out = memo[id(node)]
            return out if x._index is None else out[x._index]
        if isinstance(x, _Const):
            return x.tensor
        return x

    return tree_map(real, value)


class LazyWeight(torch.Tensor):
    """A checkpoint tensor, or a torch op applied to LazyWeights, that has not been computed."""

    @staticmethod
    def __new__(cls, meta, node, index=None):
        t = torch.Tensor._make_wrapper_subclass(
            cls,
            tuple(meta.shape),
            strides=tuple(meta.stride()),
            dtype=meta.dtype,
            layout=meta.layout,
            device=torch.device("cpu"),
            requires_grad=False,
        )
        t._node, t._index = node, index
        return t

    @classmethod
    def source(cls, tensor):
        STATE.sources += 1
        return cls(tensor, _Source(tensor))

    def recipe_digest(self) -> str:
        return self._node.digest() if self._index is None else f"{self._node.digest()}[{self._index}]"

    def __repr__(self, *, tensor_contents=None):
        return f"LazyWeight(shape={tuple(self.shape)}, dtype={self.dtype})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _SHARES_MEMORY:
            raise _taint(f"{getattr(func, '__name__', func)} exposes a checkpoint tensor's memory")
        if func in _READS_VALUES:
            return func(*materialize(list(args)), **materialize(kwargs))
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        reason = _why_not_recorded(func, args, kwargs)
        if reason is None:
            try:
                out = func(*tree_map(_meta, list(args)), **tree_map(_meta, dict(kwargs)))
            except Exception as exc:  # no meta kernel, or one that needs data
                reason = f"meta {type(exc).__name__}"
        if reason is None:
            freeze = lambda x: _Const(x) if isinstance(x, torch.Tensor) and not isinstance(x, LazyWeight) else x
            node = _Call(func, tree_map(freeze, list(args)), tree_map(freeze, dict(kwargs)))
            STATE.recorded += 1
            if isinstance(out, torch.Tensor):
                return LazyWeight(out, node)
            return type(out)(LazyWeight(m, node, i) for i, m in enumerate(out))
        return _eager(func, args, kwargs, reason)


def _eager(func, args, kwargs, reason):
    STATE.eager[f"{func}: {reason}"] += 1
    for arg, value in _args_by_schema(func, args, kwargs):
        if _is_write(arg) and _has_lazy(value):
            raise _taint(f"{func} writes into a checkpoint-derived tensor")
    memo: dict = {}
    real_args, real_kwargs = materialize(list(args), memo), materialize(dict(kwargs), memo)
    out = func(*real_args, **real_kwargs)
    lazy_real = [
        r
        for x, r in zip(tree_flatten((args, kwargs))[0], tree_flatten((real_args, real_kwargs))[0])
        if isinstance(x, LazyWeight)
    ]
    shared = {r.untyped_storage().data_ptr() for r in lazy_real if r.untyped_storage().nbytes()}
    for o in tree_flatten(out)[0]:
        if isinstance(o, torch.Tensor) and o.untyped_storage().nbytes() and o.untyped_storage().data_ptr() in shared:
            raise _taint(f"{func} returned a tensor sharing a checkpoint tensor's memory")
    return out


_SHARES_MEMORY = {
    getattr(torch.Tensor, name)
    for name in ("numpy", "data_ptr", "untyped_storage", "storage", "_typed_storage", "__array__", "__dlpack__")
    if hasattr(torch.Tensor, name)
}
_READS_VALUES = {
    getattr(torch.Tensor, name) for name in ("tolist", "__reduce_ex__", "__deepcopy__") if hasattr(torch.Tensor, name)
}


def is_lazy(value) -> bool:
    return isinstance(value, LazyWeight)


def wrap_state_dict(state_dict):
    """Make every CPU tensor in a checkpoint dict a LazyWeight source, in place; returns the dict."""
    if not isinstance(state_dict, dict):
        return state_dict
    for key, value in list(state_dict.items()):
        if (
            isinstance(value, torch.Tensor)
            and not isinstance(value, LazyWeight)
            and value.device.type == "cpu"
            and not value.requires_grad
            and value.layout == torch.strided
        ):
            state_dict[key] = LazyWeight.source(value)
    return state_dict
