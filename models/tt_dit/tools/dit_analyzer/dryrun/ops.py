# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Shape and distribution rules for the ttnn ops a DiT forward touches.

One function per op, each ending in ``recorder.emit``. This is the *shim* half
of an op registration; the analyzer half lives in
:mod:`dit_analyzer.semantics`. Phase 8 merges the two into a single ``OpSpec``
entry -- until then, adding an op means writing a rule here and a spec there.

Anything not in the tables at the bottom is recorded as an ``unregistered``
node: visible, counted, located, and never guessed at.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..ir import PARAM, Dist
from ..region import shard_chunk_size
from . import recorder
from .context import CTX
from .fused import FUSED_KERNELS
from .stubs import Enum, MeshMapper
from .tensor import Shape, Tensor, local_shape

_LAYOUTS = ("TILE_LAYOUT", "ROW_MAJOR_LAYOUT")


# -----------------------------------------------------------------------------
# argument plumbing
# -----------------------------------------------------------------------------
def _t(x, name: str = "input_tensor", args=(), kwargs=None, index: int = 0) -> Optional[Tensor]:
    """The tensor for a positional-or-keyword argument."""
    if isinstance(x, Tensor):
        return x
    kwargs = kwargs or {}
    if name in kwargs and isinstance(kwargs[name], Tensor):
        return kwargs[name]
    pool = [a for a in args if isinstance(a, Tensor)]
    return pool[index] if len(pool) > index else None


def _cluster_axis(kwargs: Dict[str, Any], op: str) -> int:
    """The mesh axis a collective runs on. Never guessed: a missing axis would
    turn a collective into an ordinary node and quietly delete it from the
    analysis."""
    axis = kwargs.get("cluster_axis")
    if axis is None:
        raise ValueError(
            "%s called without cluster_axis at %s: the dry run cannot tell which mesh "
            "axis this collective runs on" % (op, recorder.loc() or "unknown location")
        )
    return int(axis)


def _dim(kwargs: Dict[str, Any], x: Tensor, default: int = -1) -> int:
    return int(kwargs.get("dim", default)) % len(x.logical)


def _remap_dist(dist: Dist, mapping: Dict[int, Optional[int]]) -> Dist:
    return Dist(tuple(None if a is None else mapping.get(a, a) for a in dist.shard), dist.partial)


def _feature_mesh_axis(x: Tensor) -> Optional[int]:
    """Which mesh axis fractures the last (feature) axis, if any."""
    last = len(x.logical) - 1
    for m, a in enumerate(x.dist.shard):
        if a is not None and a % len(x.logical) == last:
            return m
    return None


# -----------------------------------------------------------------------------
# tensor creation
# -----------------------------------------------------------------------------
def from_torch(x, layout=None, dtype=None, device=None, mesh_mapper=None, memory_config=None, **k) -> Tensor:
    """Host tensor -> device tensor. The mesh mapper *is* the placement (blocker 8)."""
    dist = Dist.replicated(CTX.mesh)
    if mesh_mapper is not None:
        if not isinstance(mesh_mapper, MeshMapper):
            raise TypeError(
                "from_torch got a %s as mesh_mapper at %s; the dry run would read that as "
                "'replicated' and silently lose the tensor's distribution"
                % (type(mesh_mapper).__name__, recorder.loc() or "unknown location")
            )
        if mesh_mapper.dims:
            dist = Dist.make(CTX.mesh, mesh_mapper.shard())
    return recorder.entry(
        list(x.shape),
        dist,
        dtype,
        kind=PARAM if CTX.loading_weights else CTX.entry_kind,
        base=CTX.entry_base or "const",
        host=device is None,
        layout=layout,
    )


def creation(*shape, **k) -> Tensor:
    """`zeros` / `ones` / `empty`: a fresh constant, replicated unless mapped."""
    if len(shape) == 1 and isinstance(shape[0], (list, tuple, Shape)):
        shape = tuple(shape[0])
    mapper = k.get("mesh_mapper")
    dist = Dist.make(CTX.mesh, mapper.shard()) if isinstance(mapper, MeshMapper) and mapper.dims else None
    return recorder.entry(list(shape) or [1], dist or Dist.replicated(CTX.mesh), k.get("dtype"), base="const")


def to_torch(x, mesh_composer=None, **k):
    """Readback: ends the graph (blocker 43) and returns a host metadata tensor."""
    from .hostenv import host_tensor

    if not isinstance(x, Tensor):
        return x
    recorder.emit("host_read", [x], x.logical, x.dist, attrs={"devices": [0]}, base="readback")
    return host_tensor(x.logical)


def get_device_tensors(x, **k) -> List[Tensor]:
    if not isinstance(x, Tensor):
        return [x]
    local = local_shape(x.logical, x.dist)
    out = recorder.emit(
        "host_read", [x], x.logical, x.dist, attrs={"devices": [0]}, base="device_tensor", label="device 0"
    )
    return [Tensor(local, Dist.replicated(CTX.mesh), x.dtype, sym=out.sym, host=True)]


# -----------------------------------------------------------------------------
# views and elementwise
# -----------------------------------------------------------------------------
def reshape(x: Tensor, shape, **k) -> Tensor:
    """Model code reshapes the *local* view; lift the result back to logical extents.

    A rank change that only moves the non-unit axes around (``[1,N,D] ->
    [1,1,N,D]``) keeps the distribution, remapped onto the new axis positions.
    Anything else keeps the shard in place and lets the analyzer's ``identity``
    spec decide what survives -- real rank-changing reshape semantics (the VAE's
    ``b,h,w,c -> 1,1,h*w,c``) is blocker 17, phase 8.
    """
    shape = [int(d) for d in (shape if isinstance(shape, (list, tuple, Shape)) else [shape])]
    local = list(local_shape(x.logical, x.dist))
    scale = {}
    for mesh_axis, a in enumerate(x.dist.shard):
        if a is not None:
            scale[a % len(local)] = CTX.mesh.shape[mesh_axis]
    if -1 in shape:
        known = 1
        for d in shape:
            if d != -1:
                known *= d
        total = 1
        for d in local:
            total *= d
        shape[shape.index(-1)] = total // max(1, known)

    src = [(i, d) for i, d in enumerate(local) if d != 1]
    dst = [(i, d) for i, d in enumerate(shape) if d != 1]
    if [d for _, d in src] == [d for _, d in dst]:  # trivial rank change
        mapping = {i: j for (i, _), (j, _) in zip(src, dst)}
        logical = [1] * len(shape)
        for i, j in mapping.items():
            logical[j] = local[i] * scale.get(i, 1)
        for j, d in enumerate(shape):
            if d == 1:
                logical[j] = 1
        dist = _remap_dist(x.dist, {i: mapping.get(i) for i in range(len(local))})
        return recorder.emit("identity", [x], logical, dist, base="view")
    logical = [d * scale.get(i, 1) for i, d in enumerate(shape)]
    return recorder.emit("identity", [x], logical, x.dist, base="reshape")


def unsqueeze(x: Tensor, dim: int, **k) -> Tensor:
    local = list(local_shape(x.logical, x.dist))
    local.insert(dim if dim >= 0 else len(local) + dim + 1, 1)
    return reshape(x, local)


def squeeze(x: Tensor, dim: int, **k) -> Tensor:
    local = list(local_shape(x.logical, x.dist))
    if local[dim] == 1:
        local.pop(dim)
    return reshape(x, local)


def unsqueeze_to_4D(x: Tensor, **k) -> Tensor:
    local = list(local_shape(x.logical, x.dist))
    return reshape(x, [1] * (4 - len(local)) + local) if len(local) < 4 else x


def identity(x: Tensor, *a, **k) -> Tensor:
    """`to_layout` / `typecast` / `clone` / `to_memory_config`: value preserved.

    dtype and layout both arrive as bare enums, positionally or by keyword, so
    sort them out by name rather than by position.
    """
    dtype, layout = x.dtype, x.layout
    for v in list(a) + [k.get("dtype"), k.get("layout")]:
        if isinstance(v, Enum):
            if v.name in _LAYOUTS:
                layout = v
            else:
                dtype = v
    return recorder.emit("identity", [x], x.logical, x.dist, dtype=dtype, layout=layout, base="alias")


def copy(src, dst=None, **k):
    """In-place write. SSA-ified by repointing ``dst`` at a fresh symbol.

    ``ttnn.copy(src, dst)`` overwrites a buffer the caller still holds a
    reference to, which is roadmap blocker 6. Mutating the shim tensor's symbol
    keeps the SSA invariant without the caller noticing: every later read of
    ``dst`` sees the new value, and the IR records the write as a node.
    """
    if not isinstance(dst, Tensor):
        return identity(src) if isinstance(src, Tensor) else None
    out = recorder.emit("identity", [src], dst.logical, dst.dist, base="copy")
    dst.sym, dst.logical, dst.dist = out.sym, out.logical, out.dist
    return dst


def _inplace(name):
    """`multiply_` / `add_` and friends: same value rule, written into input 0."""

    def op(a, b=None, *rest, **k):
        out = pointwise(name, [a, b] + list(rest))
        if isinstance(a, Tensor):
            a.sym, a.logical, a.dist = out.sym, out.logical, out.dist
            return a
        return out

    return op


def pointwise(fn: str, ins: Sequence[Any]) -> Tensor:
    """Elementwise op; the widest input decides the output shape (broadcasting)."""
    tensors = [t for t in ins if isinstance(t, Tensor)]
    primary = max(tensors, key=lambda t: (len(t.logical), sum(t.logical)))
    return recorder.emit("pointwise", tensors, primary.logical, primary.dist, attrs={"fn": fn}, base=fn)


def slice_axis(x: Tensor, axis: int, lo: int, hi: int) -> Tensor:
    logical = list(x.logical)
    logical[axis] = hi - lo
    return recorder.emit("slice", [x], logical, x.dist, attrs={"axis": axis, "start": lo, "stop": hi}, base="slice")


def chunk(x: Tensor, count: int, dim: int = 0, **k) -> List[Tensor]:
    step = x.logical[dim] // count
    return [slice_axis(x, dim, i * step, (i + 1) * step) for i in range(count)]


def concat(tensors: Sequence[Tensor], dim: int = 0, **k) -> Tensor:
    ts = [t for t in tensors if isinstance(t, Tensor)]
    dim = dim % len(ts[0].logical)
    logical = list(ts[0].logical)
    logical[dim] = sum(t.logical[dim] for t in ts)
    return recorder.emit("concat", ts, logical, ts[0].dist, attrs={"axis": dim}, base="concat")


def permute(x: Tensor, dims, **k) -> Tensor:
    dims = [d % len(x.logical) for d in dims]
    logical = [x.logical[d] for d in dims]
    inverse = {src: dst for dst, src in enumerate(dims)}
    return recorder.emit("permute", [x], logical, _remap_dist(x.dist, inverse), attrs={"perm": dims}, base="permute")


def _binary(name):
    def op(a, b=None, *rest, **k):
        return pointwise(name, [a, b] + list(rest))

    return op


def _unary(name):
    def op(x, *a, **k):
        return pointwise(name, [x])

    return op


def addcmul(a, b=None, c=None, *rest, **k) -> Tensor:
    return pointwise("addcmul", [a, b, c])


# -----------------------------------------------------------------------------
# collectives
# -----------------------------------------------------------------------------
def all_gather_async(input_tensor=None, *a, **k) -> Tensor:
    x = _t(input_tensor, "input_tensor", a, k)
    m = _cluster_axis(k, "all_gather_async")
    return recorder.emit(
        "all_gather",
        [x],
        x.logical,
        x.dist.with_shard(m, None),
        attrs={"dim": _dim(k, x)},
        mesh_axis=m,
        base="ag",
    )


def reduce_scatter_minimal_async(input_tensor=None, *a, **k) -> Tensor:
    x = _t(input_tensor, "input_tensor", a, k)
    m = _cluster_axis(k, "reduce_scatter_minimal_async")
    dim = _dim(k, x)
    return recorder.emit(
        "reduce_scatter",
        [x],
        x.logical,
        x.dist.with_partial(m, False).with_shard(m, dim),
        attrs={"dim": dim},
        mesh_axis=m,
        base="rs",
    )


def all_reduce_async(input_tensor=None, *a, **k) -> Tensor:
    x = _t(input_tensor, "input_tensor", a, k)
    m = _cluster_axis(k, "all_reduce_async")
    return recorder.emit(
        "all_reduce", [x], x.logical, x.dist.with_partial(m, False).with_shard(m, None), mesh_axis=m, base="ar"
    )


# -----------------------------------------------------------------------------
# matmuls, fused and plain
# -----------------------------------------------------------------------------
def _matmul_dist(x: Tensor, w: Tensor) -> Dist:
    """Where the product lands: partial over a contracted axis, sharded over N."""
    xk, wk, wn = len(x.logical) - 1, len(w.logical) - 2, len(w.logical) - 1
    out = Dist.replicated(CTX.mesh)
    for m in range(len(CTX.mesh.shape)):
        xs = None if x.dist.shard[m] is None else x.dist.shard[m] % len(x.logical)
        ws = None if w.dist.shard[m] is None else w.dist.shard[m] % len(w.logical)
        if xs == xk and ws == wk:
            out = out.with_partial(m, True)
        elif ws == wn:
            out = out.with_shard(m, len(x.logical) - 1)
        elif xs is not None and xs < xk:
            out = out.with_shard(m, xs)
    return out


_CHUNK_CACHE: Dict[Tuple[str, int, int], Tensor] = {}


def _weight_chunk(w: Tensor, index: int, count: int) -> Tensor:
    """One column block of a chunked (fused) weight, as its own symbol.

    ``to_qkv(chunks=3)`` / ``to_kv(chunks=2)`` slice the weight by output columns
    with ``torch.chunk``, so the block width follows the chunk rule
    (``shard_chunk_size``): ``ceil(N/count)`` for the leading chunks and the
    remainder for the last, *not* a floor split -- an uneven fused weight would
    otherwise lose columns. Each block is its own ``[K, block]`` PARAM symbol,
    which keeps the region algebra exact and gives the chunk a distinct value
    (reusing the fused symbol was the spike's second shim bug).

    Not modelled: the block is stored per-device interleaved
    (``_interleave_heads``), so its *columns* are strided rather than a
    contiguous logical range. The analyzer reasons about a weight's shape and
    value identity, not its column ordering, so this does not affect findings;
    pinning the interleave down is the on-device conformance job (blocker 12).
    """
    if count == 1:
        return w
    key = (w.sym or "", index, count)
    if key not in _CHUNK_CACHE:
        chunk = shard_chunk_size(w.logical[1], count)
        width = min(chunk, max(0, w.logical[1] - index * chunk))  # last block is short
        _CHUNK_CACHE[key] = recorder.entry(
            [w.logical[0], width],
            w.dist,
            w.dtype,
            kind=PARAM,
            base="%s_chunk%d" % (w.sym or "w", index),
        )
    return _CHUNK_CACHE[key]


def _matmul(x: Tensor, w: Tensor, bias=None, label=None, fused_in=None, dtype=None) -> Tensor:
    logical = list(x.logical[:-1]) + [w.logical[-1]]
    ins = [x, w] + ([bias] if isinstance(bias, Tensor) else [])
    return recorder.emit(
        "matmul", ins, logical, _matmul_dist(x, w), dtype=dtype, label=label, fused_in=fused_in, base="mm"
    )


def matmul(input_tensor=None, weight_tensor=None, *a, **k) -> Tensor:
    x = _t(input_tensor, "input_tensor", a, k)
    w = _t(weight_tensor, "weight_tensor", a, k, index=1)
    return _matmul(x, w, k.get("bias") or k.get("bias_tensor"), dtype=k.get("dtype"))


def minimal_matmul_split(input_tensor=None, weight_tensor=None, *a, **k) -> List[Tensor]:
    x = _t(input_tensor, "input_tensor", a, k)
    w = _t(weight_tensor, "weight_tensor", a, k, index=1)
    chunks = int(k.get("chunks") or 1)
    return [
        _matmul(x, _weight_chunk(w, i, chunks), k.get("bias_tensor"), label="chunk%d" % i, dtype=k.get("dtype"))
        for i in range(chunks)
    ]


def _fused_epilogue(out: Tensor, k: Dict[str, Any]) -> Tensor:
    """The optional fused pointwise tail some kernels carry (``addcmul``)."""
    if isinstance(k.get("addcmul_input_tensor1"), Tensor):
        return pointwise("addcmul", [out, k["addcmul_input_tensor1"], k.get("addcmul_input_tensor2")])
    return out


def _run_gather_then_matmul(spec, input_tensor, weight_tensor, a, k) -> List[Tensor]:
    """Builder for the AGMM shape: gather the activation over cluster_axis, then
    (optionally chunked) matmul. Structure comes from ``spec`` (blocker 18)."""
    x = _t(input_tensor, "input_tensor", a, k)
    w = _t(weight_tensor, "weight_tensor", a, k, index=1)
    m = _cluster_axis(k, spec.call)
    tag = "%s@%s" % (spec.tag, recorder.loc() or "?")
    gathered = recorder.emit(
        "all_gather",
        [x],
        x.logical,
        x.dist.with_shard(m, None),
        attrs={"dim": len(x.logical) - 1},
        mesh_axis=m,
        fused_in=tag,
        base="%s_ag" % spec.tag,
    )
    chunks = int(k.get("chunks") or 1) if spec.chunked else 1
    outs = []
    for i in range(chunks):
        out = _matmul(
            gathered,
            _weight_chunk(w, i, chunks),
            k.get("bias_tensor"),
            label="%s_mm%d" % (spec.tag, i),
            fused_in=tag,
            dtype=k.get("dtype"),
        )
        outs.append(_fused_epilogue(out, k) if spec.epilogue else out)
    return outs


def _run_matmul_then_scatter(spec, input_tensor, weight_tensor, a, k):
    """Builder for the MMRS shape: matmul, then reduce-scatter the partial sums."""
    x = _t(input_tensor, "input_tensor", a, k)
    w = _t(weight_tensor, "weight_tensor", a, k, index=1)
    m = _cluster_axis(k, spec.call)
    tag = "%s@%s" % (spec.tag, recorder.loc() or "?")
    partial = _matmul(x, w, k.get("bias"), label="%s_mm" % spec.tag, fused_in=tag, dtype=k.get("dtype"))
    dim = _dim(k, partial)
    out = recorder.emit(
        "reduce_scatter",
        [partial],
        partial.logical,
        partial.dist.with_partial(m, False).with_shard(m, dim),
        attrs={"dim": dim},
        mesh_axis=m,
        fused_in=tag,
        base="%s_rs" % spec.tag,
    )
    return None, (_fused_epilogue(out, k) if spec.epilogue else out)


_FUSED_BUILDERS = {
    "gather_then_matmul": _run_gather_then_matmul,
    "matmul_then_scatter": _run_matmul_then_scatter,
}


def _fused_dispatch(call: str):
    """Bind a table-declared fused kernel to its builder (blocker 18)."""
    spec = FUSED_KERNELS[call]
    builder = _FUSED_BUILDERS[spec.order]
    return lambda input_tensor=None, weight_tensor=None, *a, **k: builder(spec, input_tensor, weight_tensor, a, k)


all_gather_minimal_matmul_async = _fused_dispatch("all_gather_minimal_matmul_async")
minimal_matmul_strided_reduce_scatter_async = _fused_dispatch("minimal_matmul_strided_reduce_scatter_async")


def dit_minimal_matmul_addcmul_fused(x, w, scalar=1.0, a1=None, a2=None, **k) -> Tensor:
    out = _matmul(x, w, k.get("bias_tensor"), dtype=k.get("dtype"))
    return pointwise("addcmul", [out, a1, a2])


def _pair(v) -> Tuple[int, int]:
    if isinstance(v, (list, tuple)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def conv2d(input_tensor=None, weight_tensor=None, bias_tensor=None, **k):
    """[B,H,W,Cin] -> [B,Hout,Wout,Cout], channel-parallel (blocker 14).

    The conv contracts the full input channels and a spatial neighbourhood, so
    it behaves like a column-parallel projection: the output's channel axis is
    fractured wherever the weight's out-channel axis (axis 0) is, and nothing is
    a partial sum. Returns ``(output, (Hout, Wout), (weight, bias))`` to match
    ttnn's ``return_output_dim=True, return_weights_and_bias=True`` tuple.
    """
    x = _t(input_tensor, "input_tensor", kwargs=k, index=0)
    w = weight_tensor if isinstance(weight_tensor, Tensor) else _t(weight_tensor, "weight_tensor", kwargs=k, index=1)
    bias = bias_tensor if isinstance(bias_tensor, Tensor) else None
    b, h, w_in = int(k["batch_size"]), int(k["input_height"]), int(k["input_width"])
    kh, kw = _pair(k["kernel_size"])
    sh, sw = _pair(k.get("stride", 1))
    ph, pw = _pair(k.get("padding", 0))
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w_in + 2 * pw - kw) // sw + 1
    channel_axis = len(x.logical) - 1  # NHWC
    out_dist = Dist(
        tuple(channel_axis if (a is not None and a % len(w.logical) == 0) else None for a in w.dist.shard),
        (False,) * len(w.dist.shard),
    )
    out = recorder.emit(
        "conv2d",
        [x, w, *([bias] if isinstance(bias, Tensor) else [])],
        [b, out_h, out_w, w.logical[0]],
        out_dist,
        attrs={"kernel": [kh, kw], "stride": [sh, sw], "padding": [ph, pw]},
        base="conv",
    )
    return out, (out_h, out_w), (w, bias)


def group_norm(input_tensor=None, **k) -> Tensor:
    """VAE group norm over NHWC channels (blocker 14).

    Channels (last axis) are TP-fractured, but each device holds whole groups
    (``num_groups`` and ``num_channels`` are both divided by the mesh factor at
    construction), and the spatial axis is not sharded -- so the reduction is
    device-local and no collective is implied. Shape- and layout-preserving,
    value-changing; modelled with the local ``pointwise`` semantics, with the
    weight/bias/mask riding along as broadcast operands.
    """
    x = _t(input_tensor, "input_tensor", kwargs=k, index=0)
    extra = [t for t in (k.get("weight"), k.get("bias"), k.get("input_mask")) if isinstance(t, Tensor)]
    return recorder.emit("group_norm", [x, *extra], x.logical, x.dist, base="gnorm")


def dit_rms_norm_unary_fused(x, weight=None, bias=None, **k) -> Tensor:
    """Local RMS norm over the last axis, with a fused unary activation.

    Unlike ``dit_fused_distributed_rmsnorm`` this is *not* distributed: it is the
    ``RMSNorm``/``LayerNorm`` used for q/k normalisation (``normalization.py:54``),
    whose reduction axis (head_dim / feature) is not sharded, so no collective is
    implied. It reduces the last axis, so the analyzer must see that axis in full
    -- modelled as ``layernorm`` with ``needs_full_axes=[-1]``. Weight/bias, when
    present, ride along as operands so their layout is checked.
    """
    extra = [t for t in (weight, bias) if isinstance(t, Tensor)]
    return recorder.emit("layernorm", [x, *extra], x.logical, x.dist, attrs={"needs_full_axes": [-1]}, base="norm")


# -----------------------------------------------------------------------------
# norms, heads, attention
# -----------------------------------------------------------------------------
def dit_fused_distributed_rmsnorm(x, mesh_axis=None, mesh_device=None, semaphore=None, **k) -> Tensor:
    """Distributed norm; with ``num_heads_per_device`` > 1 it also splits heads."""
    heads = k.get("num_heads_per_device")
    normed = recorder.emit("distributed_norm", [x, k.get("weight")], x.logical, x.dist, base="norm")
    # num_heads_per_device defaults to 1 (`normalization.py:198`), which is the
    # *no split* case. Splitting there was the spike's first shim bug and it
    # invented nine spurious findings.
    if not heads or heads == 1:
        return normed
    feature_axis = _feature_mesh_axis(x)
    b, n, f = x.logical[1], x.logical[2], x.logical[3]
    total_heads = heads * (CTX.mesh.shape[feature_axis] if feature_axis is not None else 1)
    head_dim = f // total_heads
    return recorder.emit(
        "split_heads",
        [normed],
        [b, total_heads, n, head_dim],
        _remap_dist(normed.dist, {3: 1, 2: 2, 1: 0}),
        attrs={"head_dim": head_dim},
        fused_in="fused_norm@" + (recorder.loc() or "?"),
        base="norm_heads",
    )


def nlp_create_qkv_heads(inp: Tensor, num_heads: int = 1, num_kv_heads: int = 0, **k):
    m = _feature_mesh_axis(inp)
    total = num_heads * (CTX.mesh.shape[m] if m is not None else 1)
    b, n, f = inp.logical[1], inp.logical[2], inp.logical[3]
    head_dim = f // total
    out = recorder.emit(
        "split_heads",
        [inp],
        [b, total, n, head_dim],
        _remap_dist(inp.dist, {3: 1, 2: 2, 1: 0}),
        attrs={"head_dim": head_dim},
        base="heads",
    )
    return out, None, None


def concatenate_heads(x: Tensor, **k) -> Tensor:
    b, h, n, e = x.logical
    return recorder.emit(
        "merge_heads",
        [x],
        [b, n, h * e],
        _remap_dist(x.dist, {1: 2, 2: 1, 0: 0}),
        attrs={"head_dim": e},
        base="merged",
    )


def rotary_embedding_llama(x: Tensor, cos=None, sin=None, trans_mat=None, **k) -> Tensor:
    return pointwise("rope", [x, cos, sin])


def scaled_dot_product_attention(q: Tensor, key=None, value=None, **k) -> Tensor:
    return recorder.emit("sdpa", [q, key, value, k.get("attn_mask")], q.logical, q.dist, base="sdpa")


def joint_scaled_dot_product_attention(q, key=None, value=None, aq=None, ak=None, av=None, **k):
    joint = scaled_dot_product_attention(aq, ak or key, av or value, **k) if isinstance(aq, Tensor) else None
    return scaled_dot_product_attention(q, key, value, **k), joint


def ring_joint_scaled_dot_product_attention(q, key, value, aq=None, ak=None, av=None, **k):
    """Ring SDPA: gathers K and V over cluster_axis inside the kernel."""
    m = _cluster_axis(k, "ring_joint_scaled_dot_product_attention")
    tag = "ring_sdpa@" + (recorder.loc() or "?")
    gathered = []
    for name, t in (("k", key), ("v", value)):
        gathered.append(
            recorder.emit(
                "all_gather",
                [t],
                t.logical,
                t.dist.with_shard(m, None),
                attrs={"dim": 2},
                mesh_axis=m,
                fused_in=tag,
                base="ring_%s_ag" % name,
            )
        )
    kg, vg = gathered
    out = recorder.emit("sdpa", [q, kg, vg, ak, av], q.logical, q.dist, fused_in=tag, base="ring_sdpa")
    prompt = (
        recorder.emit("sdpa", [aq, kg, vg], aq.logical, aq.dist, fused_in=tag, base="ring_sdpa_joint")
        if isinstance(aq, Tensor) and aq.logical[2] > 0
        else None
    )
    return out, prompt, None


def split_query_key_value_and_split_heads(x, num_heads: int = 1, **k):
    """Fused [B, N, 3*H*Dh] -> q, k, v each [B, H, N, Dh] (heads on axis 1).

    ``num_heads`` is the *local* head count; the fused tensor's feature axis is
    the TP-fractured ``3 * H * head_dim``, so the global head count is
    ``num_heads * tp`` and each output moves the head block onto axis 1. Emits one
    ``split_qkv_heads`` node with three outputs, which the analyzer reasons about
    directly (per-device QKV column layout, blocker 2/18).
    """
    m = _feature_mesh_axis(x)
    tp = CTX.mesh.shape[m] if m is not None else 1
    total_heads = num_heads * tp
    b, n, f = x.logical[-3], x.logical[-2], x.logical[-1]
    head_dim = f // (3 * total_heads)
    out_dist = _remap_dist(x.dist, {len(x.logical) - 1: 1, len(x.logical) - 2: 2})
    outs = [([b, total_heads, n, head_dim], out_dist) for _ in range(3)]
    return recorder.emit_multi(
        "split_qkv_heads",
        [x],
        outs,
        attrs={"heads": total_heads, "head_dim": head_dim, "qkv_layout": "per_device"},
        base="qkv",
    )


# -----------------------------------------------------------------------------
# op tables: ttnn.<name>, ttnn.experimental.<name>, ttnn.transformer.<name>
# -----------------------------------------------------------------------------
TENSOR_OPS = {
    "from_torch": from_torch,
    "to_torch": to_torch,
    "get_device_tensors": get_device_tensors,
    "zeros": creation,
    "ones": creation,
    "empty": creation,
    "full": lambda shape, value=0, **k: creation(shape, **k),
    "zeros_like": lambda x, **k: creation(list(x.shape), **k),
    "empty_like": lambda x, **k: creation(list(x.shape), **k),
    "allocate_tensor_on_device": lambda spec, device=None, **k: creation([1], **k),
    "reshape": reshape,
    "unsqueeze": unsqueeze,
    "squeeze": squeeze,
    "unsqueeze_to_4D": unsqueeze_to_4D,
    "to_layout": identity,
    "typecast": identity,
    "clone": identity,
    "to_memory_config": identity,
    "chunk": chunk,
    "concat": concat,
    "permute": permute,
    "add": _binary("add"),
    "sub": _binary("sub"),
    "subtract": _binary("sub"),
    "mul": _binary("mul"),
    "multiply": _binary("mul"),
    "div": _binary("div"),
    "lerp": lambda a, b, w, **k: pointwise("lerp", [a, b, w]),
    "addcmul": addcmul,
    "sigmoid": _unary("sigmoid"),
    "gelu": _unary("gelu"),
    "silu": _unary("silu"),
    "tanh": _unary("tanh"),
    "sqrt": _unary("sqrt"),
    "reciprocal": _unary("reciprocal"),
    "neg": _unary("neg"),
    "clamp": _unary("clamp"),
    "matmul": matmul,
    "linear": matmul,
    "conv2d": conv2d,
    "group_norm": group_norm,
    "copy": copy,
    "add_": _inplace("add"),
    "multiply_": _inplace("mul"),
    "mul_": _inplace("mul"),
    "sub_": _inplace("sub"),
}

EXPERIMENTAL_OPS = {
    "all_gather_async": all_gather_async,
    "reduce_scatter_minimal_async": reduce_scatter_minimal_async,
    "all_reduce_async": all_reduce_async,
    "minimal_matmul": matmul,
    "minimal_matmul_split": minimal_matmul_split,
    "all_gather_minimal_matmul_async": all_gather_minimal_matmul_async,
    "minimal_matmul_strided_reduce_scatter_async": minimal_matmul_strided_reduce_scatter_async,
    "dit_minimal_matmul_addcmul_fused": dit_minimal_matmul_addcmul_fused,
    "dit_rms_norm_unary_fused": dit_rms_norm_unary_fused,
    "dit_fused_distributed_rmsnorm": dit_fused_distributed_rmsnorm,
    "dit_fused_distributed_layernorm": dit_fused_distributed_rmsnorm,
    "dit_fused_distributed_rmsnorm_create_stats_buffer": lambda *a, **k: None,
    "dit_fused_distributed_layernorm_create_stats_buffer": lambda *a, **k: None,
    "nlp_create_qkv_heads": nlp_create_qkv_heads,
    "rotary_embedding_llama": rotary_embedding_llama,
    "alt_complex_rotate90": _unary("alt_complex_rotate90"),
}

TRANSFORMER_OPS = {
    "scaled_dot_product_attention": scaled_dot_product_attention,
    "joint_scaled_dot_product_attention": joint_scaled_dot_product_attention,
    "ring_joint_scaled_dot_product_attention": ring_joint_scaled_dot_product_attention,
    "exp_ring_joint_scaled_dot_product_attention": ring_joint_scaled_dot_product_attention,
    "concatenate_heads": concatenate_heads,
    "split_query_key_value_and_split_heads": split_query_key_value_and_split_heads,
}

#: Calls with no effect on the graph. Trace capture is in here on purpose: the
#: dry run replays the forward in Python every time, which is what dissolves the
#: trace-capture blocker (3) instead of solving it.
NOOPS = {
    "deallocate",
    "synchronize_device",
    "reset_global_semaphore_value",
    "dump_tensor",
    "reallocate",
    "set_fabric_config",
    "close_mesh_device",
    "ReadDeviceProfiler",
    "distributed_context_barrier",
    "begin_trace_capture",
    "end_trace_capture",
    "execute_trace",
    "release_trace",
    "DumpDeviceProfiler",
    "synchronize_devices",
}


def reset_caches() -> None:
    _CHUNK_CACHE.clear()
