# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Graph capture: record a real DiT forward pass into an analyzer graph.

Two-step flow, because the analysis itself needs no hardware:

1. **on device** -- wrap the forward pass and dump a trace::

       from dit_analyzer.capture import capture

       with capture(mesh_device, name="sd35_block") as cap:
           model(spatial, prompt, ...)
       cap.write("sd35.trace.json")

2. **offline** -- lift the trace to a graph and analyze it::

       from dit_analyzer.capture import Trace, trace_to_graph
       from dit_analyzer.ir import Dist

       trace = Trace.read("sd35.trace.json")
       print(trace.entry_summary())          # which entries need a placement
       graph = trace_to_graph(trace, placements={"in0": Dist.make(mesh, {0: 1, 1: 2})})

Why an explicit placement step: a ttnn tensor's ``.shape`` is the *per-device*
shape, and nothing on the tensor says which mesh axis fractures which tensor
axis. Every collective in the trace tells us how the layout *changes*, but the
layout of the tensors entering the traced region has to be declared. Guessing it
would make every downstream verdict unsound, so the analyzer asks instead
(``Dist.replicated`` is assumed for anything left out, and each assumption shows
up as a diagnostic).

NOTE: the recorder half of this module needs a live ttnn and has not been run on
hardware yet -- it is wired against the op names used in this tree today (see
``HOOKS``). The analysis half is exercised by the test suite.
"""

from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .ir import ACT, PARAM, Dist, Graph, Mesh, Node, Placement, TensorSymbol

# -----------------------------------------------------------------------------
# hook table: ttnn call -> canonical analyzer op
# -----------------------------------------------------------------------------
#   path, canonical op, ordered input kwarg names, attr kwargs to record
HOOKS: List[Tuple[str, str, Tuple[str, ...], Tuple[str, ...]]] = [
    ("ttnn.experimental.all_gather_async", "all_gather", ("input_tensor",), ("dim", "cluster_axis", "num_links")),
    ("ttnn.experimental.reduce_scatter_minimal_async", "reduce_scatter", ("input_tensor",), ("dim", "cluster_axis")),
    ("ttnn.experimental.all_reduce_async", "all_reduce", ("input_tensor",), ("cluster_axis",)),
    # fused: recorded as one op, expanded into stages by trace_to_graph
    (
        "ttnn.experimental.all_gather_minimal_matmul_async",
        "agmm",
        ("input_tensor", "weight_tensor", "bias_tensor"),
        ("dim", "cluster_axis", "fuse_swiglu"),
    ),
    (
        "ttnn.experimental.minimal_matmul_strided_reduce_scatter_async",
        "mmrs",
        ("input_tensor", "weight_tensor", "bias"),
        ("dim", "cluster_axis"),
    ),
    ("ttnn.experimental.minimal_matmul", "matmul", ("input_tensor", "weight_tensor", "bias_tensor"), ()),
    ("ttnn.linear", "matmul", (), ()),
    ("ttnn.matmul", "matmul", (), ()),
    ("ttnn.experimental.dit_fused_distributed_layernorm", "distributed_norm", (), ()),
    ("ttnn.experimental.dit_fused_distributed_rmsnorm", "distributed_norm", (), ()),
    ("ttnn.layer_norm", "layernorm", (), ()),
    ("ttnn.rms_norm", "layernorm", (), ()),
    (
        "ttnn.transformer.split_query_key_value_and_split_heads",
        "split_qkv_heads",
        ("input_tensor",),
        ("num_heads",),
    ),
    ("ttnn.transformer.concatenate_heads", "merge_heads", (), ()),
    (
        "ttnn.transformer.ring_joint_scaled_dot_product_attention",
        "ring_sdpa",
        (),
        ("dim", "cluster_axis", "joint_strategy"),
    ),
    ("ttnn.transformer.joint_scaled_dot_product_attention", "sdpa", (), ()),
    ("ttnn.transformer.scaled_dot_product_attention", "sdpa", (), ()),
    ("ttnn.add", "pointwise", (), ()),
    ("ttnn.mul", "pointwise", (), ()),
    ("ttnn.sub", "pointwise", (), ()),
    ("ttnn.gelu", "pointwise", (), ()),
    ("ttnn.silu", "pointwise", (), ()),
    ("ttnn.experimental.alt_complex_rotate90", "pointwise", (), ()),
    ("ttnn.reshape", "identity", (), ()),
    ("ttnn.squeeze", "identity", (), ()),
    ("ttnn.unsqueeze", "identity", (), ()),
    ("ttnn.unsqueeze_to_4D", "identity", (), ()),
    ("ttnn.to_layout", "identity", (), ()),
    ("ttnn.typecast", "identity", (), ()),
    ("ttnn.clone", "identity", (), ()),
]

TT_DIT_PREFIX = os.path.join("models", "tt_dit")


# -----------------------------------------------------------------------------
# trace format
# -----------------------------------------------------------------------------
@dataclass
class TraceOp:
    op: str
    call: str  # the ttnn function that was called
    inputs: List[int]  # trace-local tensor ids
    outputs: List[int]
    in_shapes: List[List[int]]  # per-device shapes, as ttnn reports them
    out_shapes: List[List[int]]
    dtypes: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)
    loc: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            op=self.op,
            call=self.call,
            inputs=self.inputs,
            outputs=self.outputs,
            in_shapes=self.in_shapes,
            out_shapes=self.out_shapes,
            dtypes=self.dtypes,
            attrs=self.attrs,
            loc=self.loc,
        )


@dataclass
class Trace:
    mesh_shape: Tuple[int, int]
    ops: List[TraceOp] = field(default_factory=list)
    name: str = "captured"
    steps: int = 1
    axis_names: Tuple[str, str] = ("axis0", "axis1")
    # tensor id -> generated entry name, for tensors with no traced producer
    entries: Dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            version=1,
            name=self.name,
            steps=self.steps,
            mesh_shape=list(self.mesh_shape),
            axis_names=list(self.axis_names),
            entries={str(k): v for k, v in self.entries.items()},
            ops=[o.to_dict() for o in self.ops],
        )

    def write(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @staticmethod
    def read(path: str) -> "Trace":
        with open(path) as fh:
            d = json.load(fh)
        t = Trace(
            mesh_shape=tuple(d["mesh_shape"]),
            name=d.get("name", "captured"),
            steps=d.get("steps", 1),
            axis_names=tuple(d.get("axis_names", ("axis0", "axis1"))),
            entries={int(k): v for k, v in d.get("entries", {}).items()},
        )
        for o in d["ops"]:
            t.ops.append(
                TraceOp(
                    op=o["op"],
                    call=o.get("call", o["op"]),
                    inputs=o["inputs"],
                    outputs=o["outputs"],
                    in_shapes=o["in_shapes"],
                    out_shapes=o["out_shapes"],
                    dtypes=o.get("dtypes", []),
                    attrs=o.get("attrs", {}),
                    loc=o.get("loc"),
                )
            )
        return t

    def entry_summary(self) -> str:
        """Entries whose placement the analyzer needs to be told about."""
        lines = ["entries needing a placement (pass via trace_to_graph(placements=...)):"]
        first_use: Dict[int, TraceOp] = {}
        for op in self.ops:
            for tid in op.inputs:
                if tid in self.entries and tid not in first_use:
                    first_use[tid] = op
        for tid, nm in sorted(self.entries.items(), key=lambda kv: kv[1]):
            op = first_use.get(tid)
            shape = op.in_shapes[op.inputs.index(tid)] if op else []
            lines.append("  %-8s per-device shape %-22s first used by %s" % (nm, shape, op.call if op else "?"))
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# recorder (needs ttnn)
# -----------------------------------------------------------------------------
class Recorder:
    """Records traced ttnn calls. Use via :func:`capture`."""

    def __init__(self, mesh_device, name: str = "captured", steps: int = 1, axis_names=("axis0", "axis1")):
        shape = tuple(int(x) for x in mesh_device.shape)
        self.trace = Trace(mesh_shape=shape, name=name, steps=steps, axis_names=tuple(axis_names))
        self._ids: Dict[int, int] = {}  # id(tensor) -> trace tensor id
        self._produced = set()
        self._next = 0
        self._patched: List[Tuple[Any, str, Any]] = []

    # -- tensor identity ------------------------------------------------------
    def _tid(self, tensor) -> int:
        key = id(tensor)
        if key not in self._ids:
            self._ids[key] = self._next
            self._next += 1
        return self._ids[key]

    def _register_entry(self, tid: int) -> None:
        if tid not in self._produced and tid not in self.trace.entries:
            self.trace.entries[tid] = "in%d" % len(self.trace.entries)

    # -- recording ------------------------------------------------------------
    def record(self, op: str, call: str, ins: Sequence[Any], outs: Sequence[Any], attrs: Dict[str, Any]) -> None:
        in_ids = [self._tid(t) for t in ins]
        for tid in in_ids:
            self._register_entry(tid)
        out_ids = [self._tid(t) for t in outs]
        self._produced.update(out_ids)
        self.trace.ops.append(
            TraceOp(
                op=op,
                call=call,
                inputs=in_ids,
                outputs=out_ids,
                in_shapes=[list(_shape(t)) for t in ins],
                out_shapes=[list(_shape(t)) for t in outs],
                dtypes=[_dtype(t) for t in outs],
                attrs=attrs,
                loc=_model_loc(),
            )
        )

    def write(self, path: str) -> None:
        self.trace.write(path)

    def graph(self, placements: Optional[Dict[str, Dist]] = None) -> Graph:
        return trace_to_graph(self.trace, placements=placements)


def _shape(t) -> Sequence[int]:
    return list(getattr(t, "shape", ()) or ())


def _dtype(t) -> str:
    dt = getattr(t, "dtype", None) or getattr(t, "get_dtype", lambda: None)()
    name = str(dt).lower()
    for key in ("bfloat16", "bfloat8_b", "bfloat4_b", "float32", "uint16"):
        if key in name:
            return {"bfloat16": "bf16", "bfloat8_b": "bf8_b", "bfloat4_b": "bf4_b", "float32": "fp32"}.get(key, key)
    return "bf16"


def _model_loc() -> Optional[str]:
    """Innermost stack frame inside models/tt_dit -- the line to show engineers."""
    for frame in reversed(traceback.extract_stack()[:-3]):
        if TT_DIT_PREFIX in frame.filename and "dit_analyzer" not in frame.filename:
            idx = frame.filename.find(TT_DIT_PREFIX)
            return "%s:%d" % (frame.filename[idx:], frame.lineno)
    return None


def _is_tensor(x) -> bool:
    import ttnn  # local import: only the recorder needs ttnn

    return isinstance(x, ttnn.Tensor)


def _flatten_tensors(x) -> List[Any]:
    if x is None:
        return []
    if _is_tensor(x):
        return [x]
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.extend(_flatten_tensors(item))
        return out
    return []


def _resolve(path: str):
    """Return (owner, attribute_name) for a dotted ttnn path."""
    import importlib

    parts = path.split(".")
    obj = importlib.import_module(parts[0])
    for p in parts[1:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


class capture:
    """Context manager that patches the ops in :data:`HOOKS` and records them."""

    def __init__(self, mesh_device, name: str = "captured", steps: int = 1, axis_names=("axis0", "axis1"), hooks=HOOKS):
        self.recorder = Recorder(mesh_device, name=name, steps=steps, axis_names=axis_names)
        self.hooks = hooks
        self._saved: List[Tuple[Any, str, Any]] = []

    def __enter__(self) -> Recorder:
        rec = self.recorder
        for path, op, in_names, attr_names in self.hooks:
            try:
                owner, attr = _resolve(path)
                original = getattr(owner, attr)
            except (ImportError, AttributeError):
                continue  # op not present in this build: skip quietly

            def make(original=original, op=op, path=path, in_names=in_names, attr_names=attr_names):
                def wrapper(*args, **kwargs):
                    out = original(*args, **kwargs)
                    try:
                        ins = _select_inputs(args, kwargs, in_names)
                        attrs = {k: kwargs[k] for k in attr_names if k in kwargs}
                        rec.record(op, path, ins, _flatten_tensors(out), attrs)
                    except Exception as exc:  # never break the model being traced
                        rec.trace.ops.append(
                            TraceOp(
                                op="unknown",
                                call=path,
                                inputs=[],
                                outputs=[],
                                in_shapes=[],
                                out_shapes=[],
                                dtypes=[],
                                attrs={"capture_error": repr(exc)},
                                loc=_model_loc(),
                            )
                        )
                    return out

                return wrapper

            setattr(owner, attr, make())
            self._saved.append((owner, attr, original))
        return rec

    def __exit__(self, *exc) -> bool:
        for owner, attr, original in reversed(self._saved):
            setattr(owner, attr, original)
        self._saved = []
        return False


def _select_inputs(args, kwargs, in_names: Sequence[str]) -> List[Any]:
    if in_names:
        picked = [kwargs[n] for n in in_names if n in kwargs and _is_tensor(kwargs[n])]
        if picked:
            return picked
    return _flatten_tensors(list(args)) + _flatten_tensors([kwargs[k] for k in sorted(kwargs)])


# -----------------------------------------------------------------------------
# trace -> analyzer graph (offline, no ttnn)
# -----------------------------------------------------------------------------
def logical_shape(local_shape: Sequence[int], dist: Dist, mesh: Mesh) -> Tuple[int, ...]:
    """Scale a per-device shape up to the mesh-wide logical shape."""
    shape = [int(d) for d in local_shape]
    for mesh_axis, tensor_axis in enumerate(dist.shard):
        if tensor_axis is None:
            continue
        shape[tensor_axis % len(shape)] *= mesh.size(mesh_axis)
    return tuple(shape)


def trace_to_graph(
    trace: Trace,
    placements: Optional[Dict[str, Dist]] = None,
    params: Sequence[str] = (),
    steps: Optional[int] = None,
) -> Graph:
    """Lift a recorded trace into analyzer IR.

    ``placements`` maps entry names (see :meth:`Trace.entry_summary`) to their
    mesh layout; ``params`` names the entries that are weights (constant across
    denoise steps). Anything unspecified is assumed replicated and flagged in the
    graph's ``meta['assumptions']``.
    """
    mesh = Mesh(shape=trace.mesh_shape, axis_names=trace.axis_names)
    placements = dict(placements or {})
    graph = Graph(name=trace.name, mesh=mesh, steps=steps if steps is not None else trace.steps)
    assumptions: List[str] = []

    sym_of: Dict[int, str] = {}  # trace tensor id -> symbol id
    counter = [0]

    def fresh(base: str, shape: Sequence[int], dtype: str, kind: str = ACT) -> str:
        counter[0] += 1
        sid = "%s_%d" % (base, counter[0])
        graph.symbols[sid] = TensorSymbol(id=sid, shape=tuple(shape), dtype=dtype, kind=kind, value_id=sid)
        return sid

    def entry(tid: int, local_shape: Sequence[int], dtype: str) -> str:
        name = trace.entries.get(tid, "in%d" % tid)
        dist = placements.get(name)
        if dist is None:
            dist = Dist.replicated(mesh)
            assumptions.append("%s assumed replicated (no placement given)" % name)
        kind = PARAM if name in set(params) else ACT
        sid = fresh(name, logical_shape(local_shape, dist, mesh), dtype, kind)
        graph.placements[sid] = Placement(dist=dist)
        sym_of[tid] = sid
        return sid

    def sym(tid: int, local_shape: Sequence[int], dtype: str) -> str:
        return sym_of[tid] if tid in sym_of else entry(tid, local_shape, dtype)

    def add_node(
        op: str,
        ins: Sequence[str],
        outs: Sequence[str],
        attrs: Dict[str, Any],
        top: TraceOp,
        mesh_axis=None,
        fused_in=None,
    ) -> None:
        counter[0] += 1
        graph.nodes.append(
            Node(
                id="%s_%d" % (op, counter[0]),
                op=op,
                inputs=list(ins),
                outputs=list(outs),
                attrs=dict(attrs),
                mesh_axis=mesh_axis,
                loc=top.loc,
                label=top.loc,
                fused_in=fused_in,
            )
        )

    for top in trace.ops:
        in_syms = [sym(tid, top.in_shapes[i], "bf16") for i, tid in enumerate(top.inputs)]
        dtype = top.dtypes[0] if top.dtypes else "bf16"
        in_logical = [graph.symbols[s].shape for s in in_syms]
        op = top.op
        attrs = dict(top.attrs)
        mesh_axis = attrs.pop("cluster_axis", None)
        dim = attrs.get("dim")

        if op in ("all_gather", "reduce_scatter", "all_reduce"):
            out_shape = in_logical[0]
            out = fresh(op, out_shape, dtype)
            sym_of[top.outputs[0]] = out
            add_node(op, in_syms[:1], [out], {"dim": dim if dim is not None else -1}, top, mesh_axis=mesh_axis)

        elif op == "matmul":
            out = fresh("mm", tuple(in_logical[0][:-1]) + (in_logical[1][-1],), dtype)
            sym_of[top.outputs[0]] = out
            add_node("matmul", in_syms, [out], {}, top)

        elif op == "agmm":  # gather K over the cluster axis, then matmul
            tag = "agmm@" + (top.loc or "?")
            gathered = fresh("agmm_ag", in_logical[0], dtype)
            add_node(
                "all_gather",
                in_syms[:1],
                [gathered],
                {"dim": dim if dim is not None else -1},
                top,
                mesh_axis=mesh_axis,
                fused_in=tag,
            )
            out = fresh("agmm_mm", tuple(in_logical[0][:-1]) + (in_logical[1][-1],), dtype)
            sym_of[top.outputs[0]] = out
            add_node("matmul", [gathered] + in_syms[1:], [out], {}, top, fused_in=tag)

        elif op == "mmrs":  # matmul, then reduce-scatter the partial sums
            tag = "mmrs@" + (top.loc or "?")
            partial = fresh("mmrs_mm", tuple(in_logical[0][:-1]) + (in_logical[1][-1],), dtype)
            add_node("matmul", in_syms, [partial], {}, top, fused_in=tag)
            out = fresh("mmrs_rs", graph.symbols[partial].shape, dtype)
            sym_of[top.outputs[-1]] = out
            add_node(
                "reduce_scatter",
                [partial],
                [out],
                {"dim": dim if dim is not None else -1},
                top,
                mesh_axis=mesh_axis,
                fused_in=tag,
            )

        elif op == "ring_sdpa":  # K/V gathered over the sequence axis inside the kernel
            tag = "ring_sdpa@" + (top.loc or "?")
            q, k, v = in_syms[0], in_syms[1], in_syms[2]
            kg = fresh("ring_k_ag", in_logical[1], dtype)
            add_node("all_gather", [k], [kg], {"dim": 2}, top, mesh_axis=mesh_axis, fused_in=tag)
            vg = fresh("ring_v_ag", in_logical[2], dtype)
            add_node("all_gather", [v], [vg], {"dim": 2}, top, mesh_axis=mesh_axis, fused_in=tag)
            for i, tid in enumerate(top.outputs[:2]):
                out = fresh("sdpa_out%d" % i, in_logical[0] if i == 0 else in_logical[3 + i], dtype)
                sym_of[tid] = out
                add_node("sdpa", [q if i == 0 else in_syms[3], kg, vg] + in_syms[4:], [out], {}, top, fused_in=tag)

        elif op == "split_qkv_heads":
            heads = int(attrs.get("num_heads", 1))
            head_dim = int(in_logical[0][-1]) // max(1, 3 * heads)
            outs = []
            for i, tid in enumerate(top.outputs[:3]):
                shape = (in_logical[0][0], heads, in_logical[0][1], head_dim)
                out = fresh("qkv%d" % i, shape, dtype)
                sym_of[tid] = out
                outs.append(out)
            add_node("split_qkv_heads", in_syms[:1], outs, {"heads": heads, "head_dim": head_dim}, top)

        elif op == "merge_heads":
            b, h, s, dh = in_logical[0]
            out = fresh("heads_merged", (b, s, h * dh), dtype)
            sym_of[top.outputs[0]] = out
            add_node("merge_heads", in_syms[:1], [out], {"head_dim": dh}, top)

        else:  # pointwise / identity / sdpa / unknown: shape follows the output
            for i, tid in enumerate(top.outputs):
                shape = (
                    in_logical[0] if (op != "identity" and in_logical) else _lift_rank(top.out_shapes[i], in_logical)
                )
                out = fresh(op, shape, dtype)
                sym_of[tid] = out
                add_node(op, in_syms, [out], attrs, top)

    graph.outputs = _terminal_symbols(graph)
    if assumptions:
        graph.meta["assumptions"] = assumptions
    graph.meta["captured_ops"] = len(trace.ops)
    return graph


def _lift_rank(local_shape: Sequence[int], in_logical: Sequence[Sequence[int]]) -> Tuple[int, ...]:
    """Reshape/squeeze: keep the input's logical extents, follow the new rank."""
    if not in_logical:
        return tuple(int(d) for d in local_shape)
    src = [d for d in in_logical[0] if d != 1]
    dst = list(int(d) for d in local_shape)
    non1 = [i for i, d in enumerate(dst) if d != 1]
    if len(non1) == len(src):
        for i, value in zip(non1, src):
            dst[i] = value
    return tuple(dst)


def _terminal_symbols(graph: Graph) -> List[str]:
    consumed = set()
    for n in graph.nodes:
        consumed.update(n.inputs)
    produced = [s for n in graph.nodes for s in n.outputs]
    return [s for s in produced if s not in consumed]
