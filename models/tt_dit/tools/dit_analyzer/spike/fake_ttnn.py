# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Throwaway metadata-only ``ttnn`` for the dry-run spike.

A ``Tensor`` here carries only (logical shape, distribution, dtype, layout) and
reports ``.shape`` as the **per-device** shape, the way real ttnn does — that is
what the model code branches on. Ops compute output metadata and append nodes to
an analyzer ``Graph``.

Deliberately not production code: it covers what one LTX block touches, asserts
even shard division, and ignores tiling. Its job is to answer whether the design
survives contact with `models/tt_dit`, and to count what a real shim would need.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, __file__.rsplit("/", 3)[0])  # tools/ on the path

from dit_analyzer.ir import ACT, PARAM, Dist, Graph, Mesh, Node, Placement, TensorSymbol, derive_value_id

TILE = 32

# op names encountered but not implemented here: {name: count}
UNREGISTERED: Dict[str, int] = {}


# -----------------------------------------------------------------------------
# scalars, enums, config objects
# -----------------------------------------------------------------------------
class _Enum:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return isinstance(other, _Enum) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)


class _Stub:
    """Permissive placeholder for anything that isn't a tensor."""

    def __init__(self, what: str = "stub", **kw):
        self._what = what
        self.__dict__.update(kw)

    def __call__(self, *a, **k) -> "_Stub":
        return self

    def __getattr__(self, name: str) -> "_Stub":
        return _Stub(self._what + "." + name)

    def __getitem__(self, key) -> "_Stub":
        return self

    def __iter__(self):
        return iter(())

    def __eq__(self, other) -> bool:
        return isinstance(other, _Stub) and other._what == self._what

    def __hash__(self) -> int:
        return hash(self._what)

    def __repr__(self) -> str:
        return "<%s>" % self._what


class MeshShape(tuple):
    def __new__(cls, *dims):
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = tuple(dims[0])
        return super().__new__(cls, tuple(int(d) for d in dims))


class Shape(tuple):
    """Behaves like ttnn.Shape for the operations model code performs on it."""

    def __new__(cls, dims):
        return super().__new__(cls, tuple(int(d) for d in dims))

    @property
    def rank(self) -> int:
        return len(self)

    def __eq__(self, other) -> bool:
        return tuple(self) == tuple(other) if isinstance(other, (tuple, list, Shape)) else NotImplemented

    def __ne__(self, other) -> bool:
        eq = self.__eq__(other)
        return eq if eq is NotImplemented else not eq

    def __hash__(self) -> int:
        return hash(tuple(self))


class CoreCoord:
    def __init__(self, x: int, y: int):
        self.x, self.y = x, y


class CoreGrid:
    def __init__(self, x: int = 1, y: int = 1, **k):
        self.x, self.y = x, y


class MeshDevice:
    """Metadata-only mesh device."""

    def __init__(self, shape: Sequence[int], arch_name: str = "blackhole", grid=(13, 10)):
        self.shape = MeshShape(*shape)
        self._arch = _Enum(arch_name.upper())
        self._grid = grid

    def arch(self):
        return self._arch

    def compute_with_storage_grid_size(self) -> CoreCoord:
        return CoreCoord(*self._grid)

    def core_grid(self) -> CoreGrid:
        return CoreGrid(*self._grid)

    def get_num_devices(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def create_submeshes(self, shape) -> List["MeshDevice"]:
        sub = tuple(shape)
        count = (self.shape[0] // sub[0]) * (self.shape[1] // sub[1])
        return [MeshDevice(sub, self._arch.name, self._grid) for _ in range(count)]

    def __eq__(self, other) -> bool:
        return other is self

    def __hash__(self) -> int:
        return id(self)


# -----------------------------------------------------------------------------
# tensors
# -----------------------------------------------------------------------------
_DTYPE_BYTES = {"bfloat16": "bf16", "float32": "fp32", "bfloat8_b": "bfp8_b", "bfloat4_b": "bfp4_b", "uint16": "bf16"}


class Tensor:
    __slots__ = ("logical", "dist", "dtype", "layout", "sym", "_host")

    def __init__(self, logical, dist: Dist, dtype=None, layout=None, sym: Optional[str] = None, host: bool = False):
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
    def padded_shape(self) -> Shape:
        s = list(local_shape(self.logical, self.dist))
        for a in (-2, -1):
            if len(s) >= abs(a):
                s[a] = -(-s[a] // TILE) * TILE
        return Shape(s)

    def get_dtype(self):
        return self.dtype

    def device(self):
        return None if self._host else REC.mesh_device

    def memory_config(self):
        return DRAM_MEMORY_CONFIG

    def is_allocated(self) -> bool:
        return True

    # -- ops reachable as methods -------------------------------------------
    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (list, tuple, Shape)):
            shape = tuple(shape[0])
        return reshape(self, list(shape))

    def __add__(self, other) -> "Tensor":
        return _pointwise("add", [self, other])

    __radd__ = __add__

    def __mul__(self, other) -> "Tensor":
        return _pointwise("mul", [self, other])

    __rmul__ = __mul__

    def __sub__(self, other) -> "Tensor":
        return _pointwise("sub", [self, other])

    def __getitem__(self, key) -> "Tensor":
        if not isinstance(key, tuple):
            key = (key,)
        for axis, k in enumerate(key):
            if isinstance(k, slice) and (k.start or k.stop):
                lo = k.start or 0
                hi = k.stop if k.stop is not None else self.logical[axis]
                return _slice(self, axis, lo, hi)
        return self

    def __repr__(self) -> str:
        return "FakeTTNNTensor(logical=%s, local=%s, %s)" % (list(self.logical), list(self.shape), self.dist.shard)


def local_shape(logical: Sequence[int], dist: Dist) -> Tuple[int, ...]:
    """Per-device shape: logical divided by the mesh factor on each sharded axis."""
    s = list(logical)
    for mesh_axis, tensor_axis in enumerate(dist.shard):
        if tensor_axis is None:
            continue
        a = tensor_axis % len(s)
        n = REC.mesh.shape[mesh_axis]
        assert s[a] % n == 0, "uneven shard: dim %d = %d over %d devices" % (a, s[a], n)
        s[a] //= n
    return tuple(s)


# -----------------------------------------------------------------------------
# recorder: fake ops -> analyzer IR
# -----------------------------------------------------------------------------
class Recorder:
    def __init__(self):
        self.graph: Optional[Graph] = None
        self.mesh: Optional[Mesh] = None
        self.mesh_device: Optional[MeshDevice] = None
        self._n = 0
        self.calls = 1
        self.loc_filter = "models/tt_dit"

    def start(self, mesh_device: MeshDevice, axis_names=("axis0", "axis1"), name="dryrun", steps=1, **meta) -> Graph:
        self.mesh_device = mesh_device
        self.mesh = Mesh(shape=tuple(mesh_device.shape), axis_names=axis_names, arch=mesh_device.arch().name.lower())
        self.graph = Graph(name=name, mesh=self.mesh, steps=steps, meta=dict(meta))
        self._n = 0
        UNREGISTERED.clear()
        _CHUNK_CACHE.clear()
        return self.graph

    # -- symbol / node plumbing ---------------------------------------------
    def _fresh(self, base: str) -> str:
        self._n += 1
        return "%s_%d" % (base, self._n)

    def entry(self, logical, dist: Dist, dtype=None, kind: str = ACT, base: str = "in", host: bool = False) -> Tensor:
        sid = self._fresh(base)
        self.graph.symbols[sid] = TensorSymbol(id=sid, shape=tuple(logical), dtype=_dt(dtype), kind=kind, value_id=sid)
        self.graph.placements[sid] = Placement(dist=dist)
        return Tensor(logical, dist, dtype, sym=sid, host=host)

    def emit(
        self,
        op: str,
        ins: Sequence[Tensor],
        out_logical,
        out_dist: Dist,
        attrs: Optional[Dict[str, Any]] = None,
        mesh_axis: Optional[int] = None,
        dtype=None,
        label: Optional[str] = None,
        fused_in: Optional[str] = None,
        base: Optional[str] = None,
    ) -> Tensor:
        tensors = [t for t in ins if isinstance(t, Tensor)]
        sid = self._fresh(base or op)
        self.graph.symbols[sid] = TensorSymbol(
            id=sid, shape=tuple(out_logical), dtype=_dt(dtype or (tensors[0].dtype if tensors else None)), value_id=sid
        )
        self.graph.nodes.append(
            Node(
                id=self._fresh(op + "_node"),
                op=op,
                inputs=[t.sym for t in tensors],
                outputs=[sid],
                attrs=dict(attrs or {}),
                mesh_axis=mesh_axis,
                loc=_loc(),
                label=label,
                calls=self.calls,
                fused_in=fused_in,
            )
        )
        return Tensor(out_logical, out_dist, dtype or (tensors[0].dtype if tensors else None), sym=sid)

    def unregistered(self, name: str, ins: Sequence[Any]) -> Optional[Tensor]:
        UNREGISTERED[name] = UNREGISTERED.get(name, 0) + 1
        tensors = [t for t in ins if isinstance(t, Tensor)]
        if not tensors:
            return None
        return self.emit("unknown", tensors, tensors[0].logical, tensors[0].dist, attrs={"call": name})


REC = Recorder()


def _dt(dtype) -> str:
    return _DTYPE_BYTES.get(getattr(dtype, "name", "bfloat16"), "bf16")


def _loc() -> Optional[str]:
    import traceback

    for frame in reversed(traceback.extract_stack()[:-2]):
        if REC.loc_filter in frame.filename and "dit_analyzer" not in frame.filename:
            i = frame.filename.find(REC.loc_filter)
            return "%s:%d" % (frame.filename[i:], frame.lineno)
    return None


def _t(x, name: str = "input_tensor", args=(), kwargs=None, index: int = 0) -> Tensor:
    """First tensor from a positional-or-keyword argument."""
    if isinstance(x, Tensor):
        return x
    kwargs = kwargs or {}
    if name in kwargs and isinstance(kwargs[name], Tensor):
        return kwargs[name]
    pool = [a for a in args if isinstance(a, Tensor)]
    return pool[index] if len(pool) > index else None


# -----------------------------------------------------------------------------
# dist rules (the shim half of an op registration)
# -----------------------------------------------------------------------------
def _matmul_dist(x: Tensor, w: Tensor) -> Dist:
    xk, wk = len(x.logical) - 1, len(w.logical) - 2
    wn = len(w.logical) - 1
    out = Dist.replicated(REC.mesh)
    for m in range(len(REC.mesh.shape)):
        xs = None if x.dist.shard[m] is None else x.dist.shard[m] % len(x.logical)
        ws = None if w.dist.shard[m] is None else w.dist.shard[m] % len(w.logical)
        if xs == xk and ws == wk:
            out = out.with_partial(m, True)
        elif ws == wn:
            out = out.with_shard(m, len(x.logical) - 1)
        elif xs is not None and xs < xk:
            out = out.with_shard(m, xs)
    return out


def _remap_dist(dist: Dist, mapping: Dict[int, Optional[int]]) -> Dist:
    return Dist(tuple(None if a is None else mapping.get(a, a) for a in dist.shard), dist.partial)


# -----------------------------------------------------------------------------
# ops
# -----------------------------------------------------------------------------
def from_torch(x, layout=None, dtype=None, device=None, mesh_mapper=None, memory_config=None, **k) -> Tensor:
    shape = tuple(x.shape)
    dist = Dist.replicated(REC.mesh)
    if isinstance(mesh_mapper, _MeshMapper) and mesh_mapper.dims:
        shard = {m: a for m, a in enumerate(mesh_mapper.dims) if a is not None}
        dist = Dist.make(REC.mesh, shard)
    logical = list(shape)
    return REC.entry(logical, dist, dtype, base="const", host=device is None)


def _creation(*shape, **k) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (list, tuple, Shape)):
        shape = tuple(shape[0])
    return REC.entry(list(shape) or [1], Dist.replicated(REC.mesh), k.get("dtype"), base="const")


def reshape(x: Tensor, shape, **k) -> Tensor:
    shape = [int(d) for d in (shape if isinstance(shape, (list, tuple, Shape)) else [shape])]
    # model code reshapes the *local* view; lift it back to logical extents
    local = list(local_shape(x.logical, x.dist))
    scale = {}
    for mesh_axis, a in enumerate(x.dist.shard):
        if a is not None:
            scale[a % len(local)] = REC.mesh.shape[mesh_axis]
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
        return REC.emit("identity", [x], logical, dist, base="view")
    logical = [d * scale.get(i, 1) for i, d in enumerate(shape)]
    return REC.emit("identity", [x], logical, x.dist, base="reshape")


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


def _identity(x: Tensor, *a, **k) -> Tensor:
    return REC.emit("identity", [x], x.logical, x.dist, base="alias")


def _pointwise(fn: str, ins: Sequence[Any]) -> Tensor:
    tensors = [t for t in ins if isinstance(t, Tensor)]
    primary = max(tensors, key=lambda t: (len(t.logical), sum(d for d in t.logical)))
    return REC.emit("pointwise", tensors, primary.logical, primary.dist, attrs={"fn": fn}, base=fn)


def _slice(x: Tensor, axis: int, lo: int, hi: int) -> Tensor:
    logical = list(x.logical)
    logical[axis] = hi - lo
    return REC.emit("slice", [x], logical, x.dist, attrs={"axis": axis, "start": lo, "stop": hi}, base="slice")


def chunk(x: Tensor, count: int, dim: int = 0, **k) -> List[Tensor]:
    step = x.logical[dim] // count
    return [_slice(x, dim, i * step, (i + 1) * step) for i in range(count)]


def permute(x: Tensor, dims, **k) -> Tensor:
    dims = [d % len(x.logical) for d in dims]
    logical = [x.logical[d] for d in dims]
    inverse = {src: dst for dst, src in enumerate(dims)}
    return REC.emit("permute", [x], logical, _remap_dist(x.dist, inverse), attrs={"perm": dims}, base="permute")


def all_gather_async(input_tensor=None, *a, **k) -> Tensor:
    x = _t(input_tensor, "input_tensor", a, k)
    m = k.get("cluster_axis")
    dim = k.get("dim", -1) % len(x.logical)
    return REC.emit(
        "all_gather", [x], x.logical, x.dist.with_shard(m, None), attrs={"dim": dim}, mesh_axis=m, base="ag"
    )


def reduce_scatter_minimal_async(input_tensor=None, *a, **k) -> Tensor:
    x = _t(input_tensor, "input_tensor", a, k)
    m = k.get("cluster_axis")
    dim = k.get("dim", -1) % len(x.logical)
    dist = x.dist.with_partial(m, False).with_shard(m, dim)
    return REC.emit("reduce_scatter", [x], x.logical, dist, attrs={"dim": dim}, mesh_axis=m, base="rs")


_CHUNK_CACHE: Dict[Tuple[str, int, int], "Tensor"] = {}


def _weight_chunk(w: "Tensor", index: int, count: int) -> "Tensor":
    """One column block of a chunked (fused) weight, as its own symbol.

    `to_qkv(chunks=3)` / `to_kv(chunks=2)` slice the weight by output columns, but
    the fused weight is stored per-device interleaved (`_interleave_heads`), so the
    block is not a contiguous logical column range. Modelling each block as its own
    [K, N/count] weight matches the maths and keeps the region algebra exact; a
    production shim wants a real `chunked_weight` op spec instead.
    """
    if count == 1:
        return w
    key = (w.sym or "", index, count)
    if key not in _CHUNK_CACHE:
        _CHUNK_CACHE[key] = REC.entry(
            [w.logical[0], w.logical[1] // count],
            w.dist,
            w.dtype,
            kind=PARAM,
            base="%s_chunk%d" % (w.sym or "w", index),
        )
    return _CHUNK_CACHE[key]


def _matmul(x: Tensor, w: Tensor, bias=None, label=None, fused_in=None, dtype=None) -> Tensor:
    logical = list(x.logical[:-1]) + [w.logical[-1]]
    ins = [x, w] + ([bias] if isinstance(bias, Tensor) else [])
    return REC.emit("matmul", ins, logical, _matmul_dist(x, w), dtype=dtype, label=label, fused_in=fused_in, base="mm")


def minimal_matmul(input_tensor=None, weight_tensor=None, *a, **k) -> Tensor:
    x = _t(input_tensor, "input_tensor", a, k)
    w = _t(weight_tensor, "weight_tensor", a, k, index=1)
    return _matmul(x, w, k.get("bias_tensor"), dtype=k.get("dtype"))


def minimal_matmul_split(input_tensor=None, weight_tensor=None, *a, **k) -> List[Tensor]:
    x = _t(input_tensor, "input_tensor", a, k)
    w = _t(weight_tensor, "weight_tensor", a, k, index=1)
    chunks = int(k.get("chunks", 1))
    outs = []
    for i in range(chunks):
        piece = _weight_chunk(w, i, chunks)
        outs.append(_matmul(x, piece, k.get("bias_tensor"), label="chunk%d" % i, dtype=k.get("dtype")))
    return outs


def all_gather_minimal_matmul_async(input_tensor=None, weight_tensor=None, *a, **k) -> List[Tensor]:
    """Fused: gather the activation over cluster_axis, then matmul (per chunk)."""
    x = _t(input_tensor, "input_tensor", a, k)
    w = _t(weight_tensor, "weight_tensor", a, k, index=1)
    m = k.get("cluster_axis")
    tag = "agmm@" + (_loc() or "?")
    gathered = REC.emit(
        "all_gather",
        [x],
        x.logical,
        x.dist.with_shard(m, None),
        attrs={"dim": len(x.logical) - 1},
        mesh_axis=m,
        fused_in=tag,
        base="agmm_ag",
    )
    chunks = int(k.get("chunks") or 1)
    outs = []
    for i in range(chunks):
        piece = _weight_chunk(w, i, chunks)
        out = _matmul(gathered, piece, k.get("bias_tensor"), label="agmm_mm%d" % i, fused_in=tag, dtype=k.get("dtype"))
        if isinstance(k.get("addcmul_input_tensor1"), Tensor):  # fused epilogue
            out = _pointwise("addcmul", [out, k["addcmul_input_tensor1"], k.get("addcmul_input_tensor2")])
        outs.append(out)
    return outs


def minimal_matmul_strided_reduce_scatter_async(input_tensor=None, weight_tensor=None, *a, **k):
    """Fused: matmul, then reduce-scatter the partial sums."""
    x = _t(input_tensor, "input_tensor", a, k)
    w = _t(weight_tensor, "weight_tensor", a, k, index=1)
    m = k.get("cluster_axis")
    tag = "mmrs@" + (_loc() or "?")
    partial = _matmul(x, w, k.get("bias"), label="mmrs_mm", fused_in=tag, dtype=k.get("dtype"))
    dim = k.get("dim", -1) % len(partial.logical)
    out = REC.emit(
        "reduce_scatter",
        [partial],
        partial.logical,
        partial.dist.with_partial(m, False).with_shard(m, dim),
        attrs={"dim": dim},
        mesh_axis=m,
        fused_in=tag,
        base="mmrs_rs",
    )
    if isinstance(k.get("addcmul_input_tensor1"), Tensor):
        out = _pointwise("addcmul", [out, k["addcmul_input_tensor1"], k.get("addcmul_input_tensor2")])
    return None, out


def dit_minimal_matmul_addcmul_fused(x, w, scalar=1.0, a1=None, a2=None, **k) -> Tensor:
    out = _matmul(x, w, k.get("bias_tensor"), dtype=k.get("dtype"))
    return _pointwise("addcmul", [out, a1, a2])


def dit_fused_distributed_rmsnorm(x, mesh_axis=None, mesh_device=None, semaphore=None, **k) -> Tensor:
    """Distributed norm; with num_heads_per_device it also splits heads (BHNE)."""
    heads = k.get("num_heads_per_device")
    normed = REC.emit("distributed_norm", [x, k.get("weight")], x.logical, x.dist, base="norm")
    # num_heads_per_device defaults to 1 (normalization.py:198), which is the
    # *no split* case -- splitting there was the spike's first shim bug.
    if not heads or heads == 1:
        return normed
    b, n, f = x.logical[1], x.logical[2], x.logical[3]
    total_heads = heads * REC.mesh.shape[_feature_mesh_axis(x)] if _feature_mesh_axis(x) is not None else heads
    head_dim = f // total_heads
    logical = [b, total_heads, n, head_dim]
    mapping = {3: 1, 2: 2, 1: 0}
    return REC.emit(
        "split_heads",
        [normed],
        logical,
        _remap_dist(normed.dist, mapping),
        attrs={"head_dim": head_dim},
        fused_in="fused_norm@" + (_loc() or "?"),
        base="norm_heads",
    )


dit_fused_distributed_layernorm = dit_fused_distributed_rmsnorm


def _feature_mesh_axis(x: Tensor) -> Optional[int]:
    last = len(x.logical) - 1
    for m, a in enumerate(x.dist.shard):
        if a is not None and a % len(x.logical) == last:
            return m
    return None


def nlp_create_qkv_heads(inp: Tensor, num_heads: int = 1, num_kv_heads: int = 0, **k):
    m = _feature_mesh_axis(inp)
    total = num_heads * (REC.mesh.shape[m] if m is not None else 1)
    b, n, f = inp.logical[1], inp.logical[2], inp.logical[3]
    head_dim = f // total
    logical = [b, total, n, head_dim]
    out = REC.emit(
        "split_heads",
        [inp],
        logical,
        _remap_dist(inp.dist, {3: 1, 2: 2, 1: 0}),
        attrs={"head_dim": head_dim},
        base="heads",
    )
    return out, None, None


def concatenate_heads(x: Tensor, **k) -> Tensor:
    b, h, n, e = x.logical
    logical = [b, n, h * e]
    return REC.emit(
        "merge_heads", [x], logical, _remap_dist(x.dist, {1: 2, 2: 1, 0: 0}), attrs={"head_dim": e}, base="merged"
    )


def rotary_embedding_llama(x: Tensor, cos=None, sin=None, trans_mat=None, **k) -> Tensor:
    return _pointwise("rope", [x, cos, sin])


def scaled_dot_product_attention(q: Tensor, key=None, value=None, **k) -> Tensor:
    return REC.emit("sdpa", [q, key, value, k.get("attn_mask")], q.logical, q.dist, base="sdpa")


def ring_joint_scaled_dot_product_attention(q, key, value, aq=None, ak=None, av=None, **k):
    """Ring SDPA: gathers K/V over cluster_axis inside the kernel."""
    m = k.get("cluster_axis")
    tag = "ring_sdpa@" + (_loc() or "?")
    kg = REC.emit(
        "all_gather",
        [key],
        key.logical,
        key.dist.with_shard(m, None),
        attrs={"dim": 2},
        mesh_axis=m,
        fused_in=tag,
        base="ring_k_ag",
    )
    vg = REC.emit(
        "all_gather",
        [value],
        value.logical,
        value.dist.with_shard(m, None),
        attrs={"dim": 2},
        mesh_axis=m,
        fused_in=tag,
        base="ring_v_ag",
    )
    out = REC.emit("sdpa", [q, kg, vg, ak, av], q.logical, q.dist, fused_in=tag, base="ring_sdpa")
    prompt = (
        REC.emit("sdpa", [aq, kg, vg], aq.logical, aq.dist, fused_in=tag, base="ring_sdpa_joint")
        if isinstance(aq, Tensor) and aq.logical[2] > 0
        else None
    )
    return out, prompt, None


exp_ring_joint_scaled_dot_product_attention = ring_joint_scaled_dot_product_attention


def _binary(name):
    def op(a, b=None, *rest, **k):
        return _pointwise(name, [a, b] + list(rest))

    return op


def addcmul(a, b=None, c=None, *rest, **k) -> Tensor:
    return _pointwise("addcmul", [a, b, c])


def _unary(name):
    def op(x, *a, **k):
        return _pointwise(name, [x])

    return op


class _MeshMapper:
    def __init__(self, dims=None):
        self.dims = list(dims) if dims else None


def ShardTensor2dMesh(device=None, mesh_shape=None, dims=None, **k) -> _MeshMapper:
    return _MeshMapper(dims)


def ShardTensorToMesh(device=None, dim=None, **k) -> _MeshMapper:
    return _MeshMapper([dim, None])


def ReplicateTensorToMesh(device=None, **k) -> _MeshMapper:
    return _MeshMapper(None)


# -----------------------------------------------------------------------------
# module assembly
# -----------------------------------------------------------------------------
bfloat16 = _Enum("bfloat16")
bfloat8_b = _Enum("bfloat8_b")
bfloat4_b = _Enum("bfloat4_b")
float32 = _Enum("float32")
uint16 = _Enum("uint16")
uint32 = _Enum("uint32")
int32 = _Enum("int32")
TILE_LAYOUT = _Enum("TILE_LAYOUT")
ROW_MAJOR_LAYOUT = _Enum("ROW_MAJOR_LAYOUT")
DRAM_MEMORY_CONFIG = _Stub("DRAM_MEMORY_CONFIG")
L1_MEMORY_CONFIG = _Stub("L1_MEMORY_CONFIG")

_TENSOR_OPS = {
    "from_torch": from_torch,
    "zeros": _creation,
    "ones": _creation,
    "empty": _creation,
    "full": lambda shape, value=0, **k: _creation(shape, **k),
    "empty_like": lambda x, **k: _creation(list(x.shape), **k),
    "allocate_tensor_on_device": lambda spec, device=None, **k: _creation([1], **k),
    "reshape": reshape,
    "unsqueeze": unsqueeze,
    "squeeze": squeeze,
    "unsqueeze_to_4D": unsqueeze_to_4D,
    "to_layout": _identity,
    "typecast": _identity,
    "clone": _identity,
    "to_memory_config": _identity,
    "chunk": chunk,
    "permute": permute,
    "add": _binary("add"),
    "sub": _binary("sub"),
    "subtract": _binary("sub"),
    "mul": _binary("mul"),
    "multiply": _binary("mul"),
    "div": _binary("div"),
    "lerp": lambda a, b, w, **k: _pointwise("lerp", [a, b, w]),
    "addcmul": addcmul,
    "sigmoid": _unary("sigmoid"),
    "gelu": _unary("gelu"),
    "silu": _unary("silu"),
    "tanh": _unary("tanh"),
    "sqrt": _unary("sqrt"),
    "reciprocal": _unary("reciprocal"),
    "neg": _unary("neg"),
    "clamp": _unary("clamp"),
}

_EXPERIMENTAL_OPS = {
    "all_gather_async": all_gather_async,
    "reduce_scatter_minimal_async": reduce_scatter_minimal_async,
    "minimal_matmul": minimal_matmul,
    "minimal_matmul_split": minimal_matmul_split,
    "all_gather_minimal_matmul_async": all_gather_minimal_matmul_async,
    "minimal_matmul_strided_reduce_scatter_async": minimal_matmul_strided_reduce_scatter_async,
    "dit_minimal_matmul_addcmul_fused": dit_minimal_matmul_addcmul_fused,
    "dit_fused_distributed_rmsnorm": dit_fused_distributed_rmsnorm,
    "dit_fused_distributed_layernorm": dit_fused_distributed_layernorm,
    "dit_fused_distributed_rmsnorm_create_stats_buffer": lambda *a, **k: None,
    "dit_fused_distributed_layernorm_create_stats_buffer": lambda *a, **k: None,
    "nlp_create_qkv_heads": nlp_create_qkv_heads,
    "rotary_embedding_llama": rotary_embedding_llama,
    "alt_complex_rotate90": _unary("alt_complex_rotate90"),
}

_TRANSFORMER_OPS = {
    "scaled_dot_product_attention": scaled_dot_product_attention,
    "joint_scaled_dot_product_attention": lambda q, k_, v, aq=None, ak=None, av=None, **k: (
        scaled_dot_product_attention(q, k_, v, **k),
        scaled_dot_product_attention(aq, k_, v, **k) if isinstance(aq, Tensor) else None,
    ),
    "ring_joint_scaled_dot_product_attention": ring_joint_scaled_dot_product_attention,
    "exp_ring_joint_scaled_dot_product_attention": ring_joint_scaled_dot_product_attention,
    "concatenate_heads": concatenate_heads,
    "split_query_key_value_and_split_heads": lambda x, num_heads=1, **k: (
        nlp_create_qkv_heads(x, num_heads=num_heads)[0],
        nlp_create_qkv_heads(x, num_heads=num_heads)[0],
        nlp_create_qkv_heads(x, num_heads=num_heads)[0],
    ),
}

_NOOPS = {
    "deallocate",
    "synchronize_device",
    "reset_global_semaphore_value",
    "dump_tensor",
    "reallocate",
    "set_fabric_config",
    "close_mesh_device",
    "ReadDeviceProfiler",
    "distributed_context_barrier",
    "release_trace",
    "begin_trace_capture",
    "end_trace_capture",
    "execute_trace",
    "copy",
}


class _FakeModule(types.ModuleType):
    """Explicit ops where we have semantics; loud stubs elsewhere."""

    def __init__(self, name: str, ops: Dict[str, Any]):
        super().__init__(name)
        self._ops = ops
        for k_, v in ops.items():
            setattr(self, k_, v)

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        if name in _NOOPS:
            return lambda *a, **k: None
        if name[:1].isupper():  # config object / enum constructor
            return _Stub(self.__name__ + "." + name)

        def maybe_op(*a, **k):
            args = list(a) + list(k.values())
            if any(isinstance(x, Tensor) for x in args) or any(
                isinstance(x, (list, tuple)) and any(isinstance(y, Tensor) for y in x) for x in args
            ):
                return REC.unregistered(self.__name__ + "." + name, args)
            return _Stub(self.__name__ + "." + name)

        return maybe_op


def install(mesh_shape=(4, 8), arch="blackhole") -> MeshDevice:
    """Register the fake as ``ttnn`` and return a metadata mesh device."""
    ttnn = _FakeModule("ttnn", _TENSOR_OPS)
    for name, value in globals().items():
        if name.startswith("_"):
            continue
        if isinstance(value, (_Enum, _Stub, type)) or name in ("MeshShape", "Shape", "Tensor", "MeshDevice"):
            setattr(ttnn, name, value)
    ttnn.Tensor = Tensor
    ttnn.Device = MeshDevice
    ttnn.MeshDevice = MeshDevice
    ttnn.MeshShape = MeshShape
    ttnn.Shape = Shape
    ttnn.CoreCoord = CoreCoord
    ttnn.CoreGrid = CoreGrid
    ttnn.ShardTensor2dMesh = ShardTensor2dMesh
    ttnn.ShardTensorToMesh = ShardTensorToMesh
    ttnn.ReplicateTensorToMesh = ReplicateTensorToMesh
    ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b = bfloat16, bfloat8_b, bfloat4_b
    ttnn.float32, ttnn.uint16, ttnn.uint32, ttnn.int32 = float32, uint16, uint32, int32
    ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT = TILE_LAYOUT, ROW_MAJOR_LAYOUT
    ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG = DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG
    ttnn.Topology = _Stub("Topology", Linear=_Enum("Linear"), Ring=_Enum("Ring"))
    ttnn.MathFidelity = _Stub("MathFidelity", HiFi2=_Enum("HiFi2"), HiFi4=_Enum("HiFi4"), LoFi=_Enum("LoFi"))
    ttnn.BufferType = _Stub("BufferType", DRAM=_Enum("DRAM"), L1=_Enum("L1"))
    ttnn.TensorMemoryLayout = _Stub("TensorMemoryLayout", INTERLEAVED=_Enum("INTERLEAVED"))
    ttnn.DumpTensorMode = _Stub("DumpTensorMode", LOCAL=_Enum("LOCAL"))

    experimental = _FakeModule("ttnn.experimental", _EXPERIMENTAL_OPS)
    transformer = _FakeModule("ttnn.transformer", _TRANSFORMER_OPS)
    device_mod = _FakeModule("ttnn.device", {})
    device_mod.Arch = _Stub("Arch", BLACKHOLE=_Enum("BLACKHOLE"), WORMHOLE_B0=_Enum("WORMHOLE_B0"))
    device_mod.is_blackhole = lambda *a, **k: arch == "blackhole"
    ttnn.experimental = experimental
    ttnn.transformer = transformer
    ttnn.device = device_mod
    ttnn.operations = _FakeModule("ttnn.operations", {})
    ttnn.distributed = _FakeModule("ttnn.distributed", {})

    for name, mod in (
        ("ttnn", ttnn),
        ("ttnn.experimental", experimental),
        ("ttnn.transformer", transformer),
        ("ttnn.device", device_mod),
        ("ttnn.operations", ttnn.operations),
        ("ttnn.distributed", ttnn.distributed),
    ):
        sys.modules[name] = mod

    mesh_device = MeshDevice(mesh_shape, arch)
    REC.mesh_device = mesh_device
    return mesh_device
