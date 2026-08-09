# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Small DSL for writing analyzer graphs by hand or from a capture trace.

Fused ttnn ops are expanded into their communication and compute stages
(``all_gather_minimal_matmul_async`` -> ``all_gather`` + ``matmul``) so the
communication inside a fused kernel is visible to the redundancy rules. The
stages keep a ``fused_in`` tag so reports can say "the all-gather fused inside
AGMM <x>" instead of pretending it was a separate op.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .ir import ACT, PARAM, UNREGISTERED_OP, Dist, Graph, Mesh, Node, Placement, TensorSymbol


@dataclass
class Value:
    """Handle to a tensor symbol inside a :class:`GraphBuilder`."""

    builder: "GraphBuilder"
    id: str

    @property
    def symbol(self) -> TensorSymbol:
        return self.builder.graph.symbols[self.id]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.symbol.shape

    def __repr__(self) -> str:
        return "Value(%s, %s)" % (self.id, list(self.shape))


class GraphBuilder:
    def __init__(self, name: str, mesh: Mesh, steps: int = 1, provenance: str = "hand-written", **meta):
        # Transcribed from the model source by a human (examples/, tests) unless the
        # caller says otherwise; findings rest on the transcription, not a device.
        self.graph = Graph(name=name, mesh=mesh, steps=steps, meta=dict(meta), provenance=provenance)
        self._counter = 0
        self._calls = 1
        self._loc: Optional[str] = None
        self._prefix = ""

    # -- plumbing -------------------------------------------------------------
    def _fresh(self, base: str) -> str:
        self._counter += 1
        return "%s%s_%d" % (self._prefix, base, self._counter)

    def _symbol(self, base: str, shape: Sequence[int], dtype: str, kind: str = ACT) -> Value:
        sid = self._fresh(base)
        self.graph.symbols[sid] = TensorSymbol(
            id=sid, shape=tuple(int(d) for d in shape), dtype=dtype, kind=kind, value_id=sid
        )
        return Value(self, sid)

    def _node(
        self,
        op: str,
        inputs: Sequence[Value],
        outputs: Sequence[Value],
        attrs: Optional[Dict] = None,
        mesh_axis: Optional[int] = None,
        label: Optional[str] = None,
        loc: Optional[str] = None,
        fused_in: Optional[str] = None,
    ) -> Node:
        node = Node(
            id=self._fresh(op),
            op=op,
            inputs=[v.id for v in inputs],
            outputs=[v.id for v in outputs],
            attrs=dict(attrs or {}),
            mesh_axis=mesh_axis,
            loc=loc or self._loc,
            label=(self._prefix + label) if label else None,
            calls=self._calls,
            fused_in=fused_in,
        )
        self.graph.nodes.append(node)
        return node

    @contextmanager
    def block(self, calls: int = 1, prefix: str = "", loc: Optional[str] = None):
        """Mark every node created inside as running ``calls`` times per forward."""
        old = (self._calls, self._prefix, self._loc)
        self._calls, self._prefix, self._loc = calls, prefix, loc or self._loc
        try:
            yield self
        finally:
            self._calls, self._prefix, self._loc = old

    @contextmanager
    def at(self, loc: str):
        old = self._loc
        self._loc = loc
        try:
            yield self
        finally:
            self._loc = old

    # -- inputs ---------------------------------------------------------------
    def input(
        self,
        name: str,
        shape: Sequence[int],
        shard: Optional[Dict[int, int]] = None,
        partial: Sequence[int] = (),
        dtype: str = "bf16",
        kind: str = ACT,
        step_varying: Optional[bool] = None,
    ) -> Value:
        v = self._symbol(name, shape, dtype, kind)
        self.graph.placements[v.id] = Placement(dist=Dist.make(self.graph.mesh, shard, partial))
        if step_varying is not None:
            self.graph.symbols[v.id].step_varying = step_varying
        return v

    def param(
        self, name: str, shape: Sequence[int], shard: Optional[Dict[int, int]] = None, dtype: str = "bf16"
    ) -> Value:
        return self.input(name, shape, shard=shard, dtype=dtype, kind=PARAM)

    # -- communication --------------------------------------------------------
    def all_gather(
        self,
        x: Value,
        dim: int,
        mesh_axis: int,
        label: Optional[str] = None,
        loc: Optional[str] = None,
        fused_in: Optional[str] = None,
    ) -> Value:
        y = self._symbol(label or "gathered", x.shape, x.symbol.dtype)
        self._node("all_gather", [x], [y], {"dim": dim}, mesh_axis=mesh_axis, label=label, loc=loc, fused_in=fused_in)
        return y

    def reduce_scatter(
        self,
        x: Value,
        dim: int,
        mesh_axis: int,
        label: Optional[str] = None,
        loc: Optional[str] = None,
        fused_in: Optional[str] = None,
    ) -> Value:
        y = self._symbol(label or "scattered", x.shape, x.symbol.dtype)
        self._node(
            "reduce_scatter", [x], [y], {"dim": dim}, mesh_axis=mesh_axis, label=label, loc=loc, fused_in=fused_in
        )
        return y

    def mesh_partition(self, x: Value, dim: int, mesh_axis: int, label: Optional[str] = None) -> Value:
        """Scatter a replicated tensor across ``mesh_axis`` on ``dim`` (dual of all_gather)."""
        y = self._symbol(label or "partitioned", x.shape, x.symbol.dtype)
        self._node("mesh_partition", [x], [y], {"dim": dim}, mesh_axis=mesh_axis, label=label)
        return y

    def neighbor_pad(
        self,
        x: Value,
        dims: Sequence[int],
        pad_left: Sequence[int],
        pad_right: Sequence[int],
        axes: Sequence[int],
        label: Optional[str] = None,
    ) -> Value:
        """Spatial halo: grow each padded dim by pad_left+pad_right; each device gains
        its neighbours' border rows along the matching mesh axis."""
        shape = list(x.shape)
        for d, pl, pr in zip(dims, pad_left, pad_right):
            shape[d] += pl + pr
        y = self._symbol(label or "halo", shape, x.symbol.dtype)
        attrs = {"dims": list(dims), "pad_left": list(pad_left), "pad_right": list(pad_right), "axes": list(axes)}
        self._node("neighbor_pad", [x], [y], attrs, mesh_axis=axes[0], label=label)
        return y

    def conv3d(
        self, x: Value, out_channels: int, kernel: Sequence[int], stride: Sequence[int], label: Optional[str] = None
    ) -> Value:
        """Valid 3-D conv (NTHWC): each device convs its (halo'd) spatial shard."""
        b, t, h, w, _ = x.shape

        def o(n, k, s):
            return (n - k) // s + 1

        shape = [b, o(t, kernel[0], stride[0]), o(h, kernel[1], stride[1]), o(w, kernel[2], stride[2]), out_channels]
        y = self._symbol(label or "conv3d", shape, x.symbol.dtype)
        self._node("conv3d", [x], [y], {"kernel": list(kernel), "stride": list(stride)}, label=label)
        return y

    def reduce_sum(self, x: Value, dim: int, keepdim: bool = True, label: Optional[str] = None) -> Value:
        """Sum over one axis (keepdim); a sharded axis becomes a partial sum."""
        shape = list(x.shape)
        if keepdim:
            shape[dim] = 1
        else:
            shape.pop(dim)
        y = self._symbol(label or "sum", shape, x.symbol.dtype)
        self._node("reduce_sum", [x], [y], {"dim": dim, "keepdim": keepdim}, label=label)
        return y

    def all_reduce(self, x: Value, mesh_axis: int, label: Optional[str] = None, loc: Optional[str] = None) -> Value:
        y = self._symbol(label or "reduced", x.shape, x.symbol.dtype)
        self._node("all_reduce", [x], [y], {}, mesh_axis=mesh_axis, label=label, loc=loc)
        return y

    # -- compute --------------------------------------------------------------
    def matmul(
        self,
        x: Value,
        w: Value,
        bias: Optional[Value] = None,
        label: Optional[str] = None,
        loc: Optional[str] = None,
        fused_in: Optional[str] = None,
    ) -> Value:
        out_shape = tuple(x.shape[:-1]) + (w.shape[-1],)
        y = self._symbol(label or "mm", out_shape, x.symbol.dtype)
        ins = [x, w] + ([bias] if bias is not None else [])
        self._node("matmul", ins, [y], {}, label=label, loc=loc, fused_in=fused_in)
        return y

    def pointwise(
        self,
        fn: str,
        ins: Sequence[Value],
        label: Optional[str] = None,
        out_shape: Optional[Sequence[int]] = None,
        loc: Optional[str] = None,
    ) -> Value:
        shape = out_shape if out_shape is not None else ins[0].shape
        y = self._symbol(label or fn, shape, ins[0].symbol.dtype)
        self._node("pointwise", ins, [y], {"fn": fn}, label=label, loc=loc)
        return y

    def add(self, a: Value, b: Value, label: Optional[str] = None) -> Value:
        return self.pointwise("add", [a, b], label=label or "add")

    def mul(self, a: Value, b: Value, label: Optional[str] = None) -> Value:
        return self.pointwise("mul", [a, b], label=label or "mul")

    def norm(
        self, x: Value, extra: Sequence[Value] = (), label: Optional[str] = None, loc: Optional[str] = None
    ) -> Value:
        """Local norm: needs the whole reduction axis on every device."""
        y = self._symbol(label or "normed", x.shape, x.symbol.dtype)
        self._node("layernorm", [x] + list(extra), [y], {"needs_full_axes": [-1]}, label=label, loc=loc)
        return y

    def dist_norm(
        self, x: Value, extra: Sequence[Value] = (), label: Optional[str] = None, loc: Optional[str] = None
    ) -> Value:
        """dit_fused_distributed_{layer,rms}norm: exchanges stats, keeps the layout."""
        y = self._symbol(label or "normed", x.shape, x.symbol.dtype)
        self._node("distributed_norm", [x] + list(extra), [y], {}, label=label, loc=loc)
        return y

    # -- structure ------------------------------------------------------------
    def view(self, x: Value, shape: Sequence[int], label: Optional[str] = None) -> Value:
        y = self._symbol(label or "view", shape, x.symbol.dtype)
        self._node("identity", [x], [y], {}, label=label)
        return y

    def slice(self, x: Value, axis: int, start: int, stop: int, label: Optional[str] = None) -> Value:
        shape = list(x.shape)
        shape[axis % len(shape)] = stop - start
        y = self._symbol(label or "slice", shape, x.symbol.dtype)
        self._node("slice", [x], [y], {"axis": axis, "start": start, "stop": stop}, label=label)
        return y

    def concat(self, ins: Sequence[Value], axis: int, label: Optional[str] = None) -> Value:
        shape = list(ins[0].shape)
        axis_n = axis % len(shape)
        shape[axis_n] = sum(v.shape[axis_n] for v in ins)
        y = self._symbol(label or "concat", shape, ins[0].symbol.dtype)
        self._node("concat", ins, [y], {"axis": axis}, label=label)
        return y

    # -- attention ------------------------------------------------------------
    def split_qkv_heads(
        self,
        qkv: Value,
        heads: int,
        head_dim: int,
        kv_heads: Optional[int] = None,
        qkv_layout: str = "per_device",
        label: Optional[str] = None,
    ) -> Tuple[Value, Value, Value]:
        """Fused QKV -> q, k, v. ``kv_heads`` (< ``heads``) makes it grouped-query."""
        kv = kv_heads if kv_heads is not None else heads
        b, s = qkv.shape[0], qkv.shape[1]
        counts = (heads, kv, kv)
        outs = [
            self._symbol((label or "qkv") + "_" + n, [b, counts[i], s, head_dim], qkv.symbol.dtype)
            for i, n in enumerate(("q", "k", "v"))
        ]
        attrs = {"heads": heads, "head_dim": head_dim, "qkv_layout": qkv_layout}
        if kv_heads is not None:
            attrs["kv_heads"] = kv
        self._node("split_qkv_heads", [qkv], outs, attrs, label=label)
        return outs[0], outs[1], outs[2]

    def embedding(self, ids: Value, weight: Value, label: Optional[str] = None) -> Value:
        """Token-id lookup [.., S] x [V, H] -> [.., S, H]."""
        y = self._symbol(label or "embed", list(ids.shape) + [weight.shape[-1]], weight.symbol.dtype)
        self._node("embedding", [ids, weight], [y], {}, label=label)
        return y

    def split_heads(self, x: Value, heads: int, head_dim: int, label: Optional[str] = None) -> Value:
        """nlp_create_qkv_heads / fused norm+split: [1, N, H*Dh] -> [1, H, N, Dh]."""
        b, s = x.shape[0], x.shape[1]
        y = self._symbol(label or "heads", [b, heads, s, head_dim], x.symbol.dtype)
        self._node("split_heads", [x], [y], {"head_dim": head_dim}, label=label)
        return y

    def merge_heads(self, x: Value, label: Optional[str] = None) -> Value:
        b, h, s, dh = x.shape
        y = self._symbol(label or "heads_merged", [b, s, h * dh], x.symbol.dtype)
        self._node("merge_heads", [x], [y], {"head_dim": dh}, label=label)
        return y

    def permute(self, x: Value, perm: Sequence[int], label: Optional[str] = None) -> Value:
        shape = [x.shape[p % len(x.shape)] for p in perm]
        y = self._symbol(label or "permuted", shape, x.symbol.dtype)
        self._node("permute", [x], [y], {"perm": list(perm)}, label=label)
        return y

    def sdpa(
        self,
        q: Value,
        k: Value,
        v: Value,
        extra: Sequence[Value] = (),
        label: Optional[str] = None,
        loc: Optional[str] = None,
    ) -> Value:
        y = self._symbol(label or "attn", q.shape, q.symbol.dtype)
        self._node("sdpa", [q, k, v] + list(extra), [y], {}, label=label, loc=loc)
        return y

    def ring_sdpa(
        self,
        q: Value,
        k: Value,
        v: Value,
        sp_axis: int,
        extra: Sequence[Value] = (),
        label: Optional[str] = None,
        loc: Optional[str] = None,
    ) -> Value:
        """ring_joint_scaled_dot_product_attention: gathers K/V over the SP axis internally."""
        tag = "ring_sdpa:" + (label or "attn")
        kg = self.all_gather(k, dim=2, mesh_axis=sp_axis, label=(label or "attn") + "_k_ag", loc=loc, fused_in=tag)
        vg = self.all_gather(v, dim=2, mesh_axis=sp_axis, label=(label or "attn") + "_v_ag", loc=loc, fused_in=tag)
        return self.sdpa(q, kg, vg, extra=extra, label=label, loc=loc)

    # -- fused ttnn ops -------------------------------------------------------
    def agmm(
        self,
        x: Value,
        w: Value,
        mesh_axis: int,
        bias: Optional[Value] = None,
        dim: int = -1,
        label: Optional[str] = None,
        loc: Optional[str] = None,
    ) -> Value:
        """all_gather_minimal_matmul_async: gather K across ``mesh_axis``, then matmul."""
        tag = "agmm:" + (label or "mm")
        xg = self.all_gather(x, dim=dim, mesh_axis=mesh_axis, label=(label or "mm") + "_ag", loc=loc, fused_in=tag)
        return self.matmul(xg, w, bias=bias, label=label, loc=loc, fused_in=tag)

    def agmm_chunks(
        self,
        x: Value,
        weights: Sequence[Value],
        mesh_axis: int,
        dim: int = -1,
        labels: Optional[Sequence[str]] = None,
        label: Optional[str] = None,
        loc: Optional[str] = None,
    ) -> List[Value]:
        """``all_gather_minimal_matmul_async(chunks=n)``: one gather, n output chunks.

        Modelled as one gather feeding one matmul per chunk, which is what the
        chunked kernel computes (the weight is one tensor split by output column
        blocks).
        """
        tag = "agmm:" + (label or "mm")
        xg = self.all_gather(x, dim=dim, mesh_axis=mesh_axis, label=(label or "mm") + "_ag", loc=loc, fused_in=tag)
        names = list(labels or ["%s_c%d" % (label or "mm", i) for i in range(len(weights))])
        return [self.matmul(xg, w, label=names[i], loc=loc, fused_in=tag) for i, w in enumerate(weights)]

    def matmul_rs(
        self,
        x: Value,
        w: Value,
        mesh_axis: int,
        bias: Optional[Value] = None,
        dim: int = -1,
        label: Optional[str] = None,
        loc: Optional[str] = None,
    ) -> Value:
        """minimal_matmul_strided_reduce_scatter_async: matmul then reduce-scatter."""
        tag = "mmrs:" + (label or "mm")
        y = self.matmul(x, w, bias=bias, label=label, loc=loc, fused_in=tag)
        return self.reduce_scatter(
            y, dim=dim, mesh_axis=mesh_axis, label=(label or "mm") + "_rs", loc=loc, fused_in=tag
        )

    def host_read(
        self, x: Value, devices: Sequence[int], label: Optional[str] = None, loc: Optional[str] = None
    ) -> Value:
        """Only ``devices`` are read downstream (e.g. ``get_device_tensors()[0]``)."""
        y = self._symbol(label or "readback", x.shape, x.symbol.dtype)
        self._node("host_read", [x], [y], {"devices": list(devices)}, label=label, loc=loc)
        return y

    # -- escape hatch ---------------------------------------------------------
    def unregistered(
        self,
        call: str,
        ins: Sequence[Value],
        out_shape: Optional[Sequence[int]] = None,
        label: Optional[str] = None,
        loc: Optional[str] = None,
    ) -> Value:
        """A ttnn call with no semantics, as the dry run records it.

        Its output metadata is *assumed* equal to input 0's, so the analyzer
        withholds any finding whose proof passes through it rather than reporting
        a claim built on a guess.
        """
        shape = out_shape if out_shape is not None else (ins[0].shape if ins else [1])
        y = self._symbol(label or "unreg", shape, ins[0].symbol.dtype if ins else "bf16")
        self._node(
            UNREGISTERED_OP,
            ins,
            [y],
            {"call": call, "arity": len(ins), "output_metadata": "assumed equal to input 0"},
            label=label,
            loc=loc,
        )
        return y

    def unknown(self, op: str, ins: Sequence[Value], out_shape: Sequence[int], label: Optional[str] = None) -> Value:
        y = self._symbol(label or "opaque", out_shape, ins[0].symbol.dtype if ins else "bf16")
        self._node(op, ins, [y], {}, label=label)
        return y

    # -- finish ---------------------------------------------------------------
    def finish(self, outputs: Sequence[Value]) -> Graph:
        self.graph.outputs = [v.id for v in outputs]
        return self.graph
