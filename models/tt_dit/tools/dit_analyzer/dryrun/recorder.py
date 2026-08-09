# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Turning shim ops into analyzer IR.

Every op in :mod:`.ops` ends in :func:`emit`, which mints a fresh SSA symbol,
appends a node, and records where in the model source the call came from. Every
tensor that enters the graph from the host goes through :func:`entry`, which is
where a distribution becomes known without anyone declaring it.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..ir import ACT, UNREGISTERED_OP, Dist, Graph, Mesh, Node, Placement, TensorSymbol, is_model_frame
from .context import CTX, DISPATCH_FILES, MODEL_MARKER, STACK_DEPTH
from .tensor import Tensor, dtype_tag


def start(mesh_device, axis_names=("axis0", "axis1"), name="dryrun", steps=1, calls=1, **meta) -> Graph:
    """Begin recording. Returns the graph the run will fill in."""
    CTX.reset()
    CTX.mesh_device = mesh_device
    CTX.mesh = Mesh(
        shape=tuple(mesh_device.shape),
        axis_names=tuple(axis_names),
        arch=mesh_device.arch().name.lower(),
        topology=str(meta.pop("topology", "Linear")),
    )
    # Shim-derived shapes -> "the shim believes" until on-device conformance.
    CTX.graph = Graph(name=name, mesh=CTX.mesh, steps=steps, meta=dict(meta), provenance="dry-run")
    CTX.calls = calls
    return CTX.graph


def fresh(base: str) -> str:
    CTX.counter += 1
    return "%s_%d" % (base, CTX.counter)


# -----------------------------------------------------------------------------
# source attribution (roadmap blocker 44)
# -----------------------------------------------------------------------------
def caller_stack(depth: int = STACK_DEPTH, hard_max: int = 8) -> List[str]:
    """tt_dit frames, innermost first, at most one per file, that reach the model call site.

    A finding that names only the innermost frame names shared library code: the spike's
    duplicate gathers landed on ``layers/linear.py:250`` (the AGMM inside ``ColParallelLinear``)
    rather than the ``to_qkv`` call that chose to gather. Both are true; only the caller is
    actionable, so ``source_chain`` leads with the model frame and lists the library frames
    underneath.

    Keep up to ``depth`` frames for that library context, but **always keep walking (to
    ``hard_max``) until a model call site is captured** — a row-parallel feedforward goes
    feedforward → linear → manager, three library frames, so a flat ``depth`` of 3 stopped
    before ever reaching the transformer block that owns the feedforward (the frame the
    engineer actually wants). If no model frame exists on the stack, stop at ``depth``.

    ``sys._getframe`` rather than ``traceback`` because this runs once per node and
    ``traceback`` reads source files to build its lines.
    """
    frames: List[str] = []
    previous_file = None
    saw_model = False
    frame = sys._getframe(1)
    while frame is not None and len(frames) < hard_max:
        filename = frame.f_code.co_filename
        cut = filename.find(MODEL_MARKER)
        if cut >= 0 and "dit_analyzer" not in filename and filename != previous_file:
            relative = filename[cut:]
            if not relative.startswith(DISPATCH_FILES):  # Module.__call__ -> forward
                model = is_model_frame(relative)
                # take library frames up to `depth`; keep taking (only the model frame) past
                # `depth` until the model call site is in, then stop.
                if len(frames) < depth or (model and not saw_model):
                    frames.append("%s:%d" % (relative, frame.f_lineno))
                    previous_file = filename
                    saw_model = saw_model or model
                if len(frames) >= depth and saw_model:
                    break
        frame = frame.f_back
    return frames


def loc() -> Optional[str]:
    """The innermost model source line, i.e. the ttnn call site itself."""
    stack = caller_stack(1)
    return stack[0] if stack else None


# -----------------------------------------------------------------------------
# symbols and nodes
# -----------------------------------------------------------------------------
def entry(
    logical,
    dist: Dist,
    dtype=None,
    kind: str = ACT,
    base: str = "in",
    host: bool = False,
    layout=None,
    step_varying: Optional[bool] = None,
) -> Tensor:
    """A tensor entering the graph from the host, with its placement recorded.

    ``step_varying=False`` declares that this input is the same on every denoise step (the
    prompt embeds, the rope tables); leaving it unset reads as varying. See TensorSymbol.
    """
    graph = CTX.require_graph()
    sid = fresh(base)
    graph.symbols[sid] = TensorSymbol(
        id=sid, shape=tuple(logical), dtype=dtype_tag(dtype), kind=kind, value_id=sid, step_varying=step_varying
    )
    graph.placements[sid] = Placement(dist=dist)
    return Tensor(logical, dist, dtype, sym=sid, host=host, layout=layout)


def emit(
    op: str,
    ins: Sequence[Any],
    out_logical,
    out_dist: Dist,
    attrs: Optional[Dict[str, Any]] = None,
    mesh_axis: Optional[int] = None,
    dtype=None,
    layout=None,
    label: Optional[str] = None,
    fused_in: Optional[str] = None,
    base: Optional[str] = None,
) -> Tensor:
    """Record one IR node and return the metadata tensor it produces."""
    graph = CTX.require_graph()
    tensors = [t for t in ins if isinstance(t, Tensor)]
    out_dtype = dtype or (tensors[0].dtype if tensors else None)
    sid = fresh(base or op)
    graph.symbols[sid] = TensorSymbol(id=sid, shape=tuple(out_logical), dtype=dtype_tag(out_dtype), value_id=sid)
    stack = caller_stack()
    graph.nodes.append(
        Node(
            id=fresh(op + "_node"),
            op=op,
            inputs=[t.sym for t in tensors],
            outputs=[sid],
            attrs=dict(attrs or {}),
            mesh_axis=mesh_axis,
            loc=stack[0] if stack else None,
            stack=stack,
            label=label,
            calls=CTX.calls,
            fused_in=fused_in,
        )
    )
    return Tensor(out_logical, out_dist, out_dtype, layout=layout or (tensors[0].layout if tensors else None), sym=sid)


def emit_multi(
    op: str,
    ins: Sequence[Any],
    outs: Sequence[Tuple[Sequence[int], Dist]],
    attrs: Optional[Dict[str, Any]] = None,
    mesh_axis: Optional[int] = None,
    dtype=None,
    label: Optional[str] = None,
    fused_in: Optional[str] = None,
    base: Optional[str] = None,
) -> List[Tensor]:
    """Record one IR node with several outputs (e.g. fused QKV split -> q, k, v).

    ``outs`` is one ``(logical, dist)`` per output. Every output is a fresh SSA
    symbol on the same node, so the analyzer spec sees them as ``node.outputs``.
    """
    graph = CTX.require_graph()
    tensors = [t for t in ins if isinstance(t, Tensor)]
    out_dtype = dtype or (tensors[0].dtype if tensors else None)
    sids, results = [], []
    for logical, dist in outs:
        sid = fresh(base or op)
        graph.symbols[sid] = TensorSymbol(id=sid, shape=tuple(logical), dtype=dtype_tag(out_dtype), value_id=sid)
        sids.append(sid)
        results.append(Tensor(logical, dist, out_dtype, layout=tensors[0].layout if tensors else None, sym=sid))
    stack = caller_stack()
    graph.nodes.append(
        Node(
            id=fresh(op + "_node"),
            op=op,
            inputs=[t.sym for t in tensors],
            outputs=sids,
            attrs=dict(attrs or {}),
            mesh_axis=mesh_axis,
            loc=stack[0] if stack else None,
            stack=stack,
            label=label,
            calls=CTX.calls,
            fused_in=fused_in,
        )
    )
    return results


def unregistered(call: str, args: Sequence[Any]) -> Optional[Tensor]:
    """An op with no spec: recorded in full, never guessed at.

    The node keeps real inputs, a real output symbol, real shapes and a real
    source location, so ``ditcheck ops --missing`` can list everything the run
    touched in one pass. Its output metadata is *assumed* to match input 0, which
    is why the analyzer refuses to emit any finding whose proof passes through
    such a node (see :mod:`dit_analyzer.semantics`, ``unregistered``).
    """
    tensors = [t for t in args if isinstance(t, Tensor)]
    where = loc()
    record = CTX.unregistered.setdefault(call, [0, where, 0])
    record[0] += 1
    record[2] = len(tensors)
    if not tensors:
        return None
    x = tensors[0]
    return emit(
        UNREGISTERED_OP,
        tensors,
        x.logical,
        x.dist,
        attrs={"call": call, "arity": len(tensors), "output_metadata": "assumed equal to input 0"},
        base="unreg",
    )


def missing_ops() -> List[Dict[str, Any]]:
    """What ``ops --missing`` reports: call name, count, arity, one call site."""
    return [
        {"call": call, "count": count, "loc": where, "tensor_args": arity}
        for call, (count, where, arity) in sorted(CTX.unregistered.items(), key=lambda kv: -kv[1][0])
    ]
