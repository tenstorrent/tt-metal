# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Distributed dispatch layer.

Every op call that has been registered flows through ``dispatch()``.  Dispatch
checks whether inputs carry distributed layouts, looks up a sharding rule,
redistributes inputs as needed, calls the raw C++ op, applies post-collectives,
and stamps the output layout.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Tuple

from .layout import Layout, Replicate, get_layout, set_layout, replicated_layout
from .mesh_runtime import get_runtime
from .redistribute import redistribute
from .debug import TraceEntry, dispatch_trace
from .rules.registry import ShardingPlan, get_rule
from .utils import is_distributed

import ttml


# ---------------------------------------------------------------------------
# Raw-op registry: keeps original (unwrapped) callables
# ---------------------------------------------------------------------------

_RAW_OPS: Dict[str, Callable] = {}


def _get_raw(op_name: str) -> Callable:
    return _RAW_OPS[op_name]


# ---------------------------------------------------------------------------
# Plan cache helpers
# ---------------------------------------------------------------------------


def _hashable_kwargs(kwargs: Dict[str, Any]) -> Tuple:
    """Convert kwargs to a hashable tuple for cache keys (best-effort)."""
    items = []
    for k, v in sorted(kwargs.items()):
        try:
            hash(v)
            items.append((k, v))
        except TypeError:
            items.append((k, id(v)))
    return tuple(items)


# ---------------------------------------------------------------------------
# Fallback: gather all inputs to replicated, run the op, return replicated
# ---------------------------------------------------------------------------


def _fallback_replicated(op_name: str, tensor_args, other_args, kwargs):
    """Gather every distributed tensor to replicated, run the raw op, return replicated output."""
    runtime = get_runtime()
    ndim = (
        len(runtime.mesh_shape)
        if runtime and hasattr(runtime.mesh_shape, "__len__")
        else 2
    )
    rep = replicated_layout(ndim)

    gathered = []
    redistributions = []
    for arg in tensor_args:
        cur = get_layout(arg)
        if cur is not None and cur != rep:
            redistributions.append({"from": cur, "to": rep})
            gathered.append(redistribute(arg, rep))
        else:
            gathered.append(arg)

    raw = _get_raw(op_name)
    result = raw(*gathered, *other_args, **kwargs)

    if hasattr(result, "get_value"):
        set_layout(result, rep)

    if dispatch_trace.enabled:
        dispatch_trace.record(
            TraceEntry(
                op_name=op_name,
                input_layouts=[get_layout(a) for a in tensor_args],
                rule_name=None,
                plan=None,
                redistributions=redistributions,
                post_collectives=[],
                output_layout=rep,
            )
        )
    return result


# ---------------------------------------------------------------------------
# dispatch()
# ---------------------------------------------------------------------------


def dispatch(op_name: str, *args, **kwargs):
    """Central dispatch entry point.

    1. Fast path: no distributed tensors → call raw op directly.
    2. Look up sharding rule.
    3. Compute (or cache) a ShardingPlan.
    4. Redistribute inputs.
    5. Call raw op on local shards.
    6. Apply post-collective if needed.
    7. Stamp output layout.
    """
    runtime = get_runtime()

    tensor_args = []
    other_args = []
    for a in args:
        if hasattr(a, "get_value"):
            tensor_args.append(a)
        else:
            other_args.append(a)

    # Check if any tensor has a layout set (meaning it's distributed)
    has_distributed = any(get_layout(t) is not None for t in tensor_args)
    if not has_distributed:
        raw = _get_raw(op_name)
        return raw(*args, **kwargs)

    input_layouts = [get_layout(t) for t in tensor_args]

    rule_fn = get_rule(op_name)
    if rule_fn is None:
        return _fallback_replicated(op_name, tensor_args, other_args, kwargs)

    cache = runtime.plan_cache
    cache_key = (op_name, tuple(input_layouts), _hashable_kwargs(kwargs))
    plan: Optional[ShardingPlan] = cache.get(cache_key)
    if plan is None:
        plan = rule_fn(*input_layouts, runtime=runtime, **kwargs)
        cache.put(cache_key, plan)

    # Apply pre-collective (e.g., broadcast for column-parallel)
    pre_collectives = []
    preprocessed = []
    layout_idx = 0
    for a in args:
        if hasattr(a, "get_value"):
            processed = a
            # Apply pre-collective to first tensor input (activation)
            if (
                layout_idx == 0
                and plan.pre_collective == "broadcast"
                and plan.pre_collective_mesh_axis is not None
            ):
                processed = ttml.ops.distributed.broadcast(
                    processed,
                    cluster_axis=plan.pre_collective_mesh_axis,
                )
                pre_collectives.append(
                    {
                        "type": "broadcast",
                        "mesh_axis": plan.pre_collective_mesh_axis,
                    }
                )
            preprocessed.append(processed)
            layout_idx += 1
        else:
            preprocessed.append(a)

    # Redistribute inputs to required layouts
    redistributions = []
    redistributed = []
    layout_idx = 0
    for a in preprocessed:
        if hasattr(a, "get_value"):
            target = plan.input_layouts[layout_idx]
            cur = input_layouts[layout_idx]
            if cur is not None and cur != target:
                redistributions.append(
                    {
                        "arg_idx": layout_idx,
                        "from": cur,
                        "to": target,
                    }
                )
                redistributed.append(
                    redistribute(a, target, grad_replicated=plan.gather_grad_replicated)
                )
            else:
                redistributed.append(a)
            layout_idx += 1
        else:
            redistributed.append(a)

    raw = _get_raw(op_name)
    result = raw(*redistributed, **kwargs)

    # Apply post-collective (e.g., all_reduce for row-parallel)
    post_collectives = []
    if plan.post_collective == "all_reduce" and plan.reduce_mesh_axis is not None:
        result = ttml.ops.distributed.all_reduce(
            result,
            noop_backward=plan.noop_backward,
            cluster_axis=plan.reduce_mesh_axis,
        )
        post_collectives.append(
            {
                "type": "all_reduce",
                "mesh_axis": plan.reduce_mesh_axis,
                "noop_backward": plan.noop_backward,
            }
        )

    if hasattr(result, "get_value"):
        set_layout(result, plan.output_layout)

    if dispatch_trace.enabled:
        dispatch_trace.record(
            TraceEntry(
                op_name=op_name,
                input_layouts=input_layouts,
                rule_name=rule_fn.__name__ if rule_fn else None,
                plan=plan,
                redistributions=redistributions,
                post_collectives=post_collectives,
                output_layout=plan.output_layout,
            )
        )

    return result


# ---------------------------------------------------------------------------
# register_op decorator: wraps a raw callable into dispatch
# ---------------------------------------------------------------------------


def register_op(op_name: str, target: Callable):
    """Register an op for distributed dispatch.

    Saves the raw callable and returns a wrapper that routes through
    ``dispatch(op_name, ...)``.

    This is the decorator that ``_register_ops`` uses at import-time to
    monkey-patch ``ttml.ops.*`` entry points.
    """
    _RAW_OPS[op_name] = target

    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        return dispatch(op_name, *args, **kwargs)

    return wrapper
