# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Distributed dispatch layer.

Every op call that has been registered flows through ``dispatch()``.  Dispatch
checks whether inputs carry distributed layouts, looks up a sharding rule,
redistributes inputs as needed, calls the raw C++ op, optionally applies
pre- and post-collectives, and stamps the output layout.

Pre- and post-collectives (see ShardingPlan in rules.registry):
    Optional CCLs per input and per output. They do not change tensor shape
    (broadcast, all_reduce); the op would run without them. The rule specifies
    pre_collectives[i] for each tensor input (None = no collective) and
    post_collectives[j] for each output (None = no collective). Supported:
    pre: {"type": "broadcast", "mesh_axis": int}; post: {"type": "all_reduce", ...}
    or {"type": "all_gather", ...} (all_gather may change shape, e.g. LM head).
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Tuple

from .layout import get_layout, set_layout, replicated_layout
from .mesh_runtime import get_runtime
from .redistribute import redistribute
from .debug import TraceEntry, dispatch_trace
from .rules.registry import ShardingPlan, CCL, get_rule
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


_COLLECTIVE_OPS_DIM_CLUSTER = frozenset({"scatter", "all_gather", "reduce_scatter"})


def _rule_kwargs_for_op(
    op_name: str, other_args: List[Any], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge leading non-tensor args into kwargs for collectives that take ``dim`` / ``cluster_axis``.

    Non-tensor positionals are still forwarded to ``raw`` via ``*other_args``; this copy
    is for plan cache keys and sharding rules only.
    """
    merged = dict(kwargs)
    if op_name not in _COLLECTIVE_OPS_DIM_CLUSTER:
        return merged
    if "dim" not in merged and len(other_args) >= 1:
        merged["dim"] = other_args[0]
    if "cluster_axis" not in merged and len(other_args) >= 2:
        merged["cluster_axis"] = other_args[1]
    return merged


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

    if isinstance(result, list):
        for r in result:
            set_layout(r, rep)
    else:
        set_layout(result, rep)

    if dispatch_trace.enabled:
        dispatch_trace.record(
            TraceEntry(
                op_name=op_name,
                input_layouts=[get_layout(a) for a in tensor_args],
                rule_name=None,
                plan=None,
                pre_collectives=[],
                redistributions=redistributions,
                post_collectives=[],
                output_layout=rep,
                op_kwargs=dict(kwargs),
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
    dispatch_trace.enter_dispatch()
    runtime = get_runtime()

    tensor_args = []
    other_args = []
    for a in args:
        if isinstance(a, ttml.autograd.Tensor):
            tensor_args.append(a)
        else:
            other_args.append(a)

    # Check if any tensor is actually distributed (lives on a mesh device)
    has_distributed = any(is_distributed(t) for t in tensor_args)
    if not has_distributed:
        raw = _get_raw(op_name)
        result = raw(*args, **kwargs)
        # Record fast-path when tracing is enabled so it's visible in debug output
        if dispatch_trace.enabled:
            dispatch_trace.record(
                TraceEntry(
                    op_name=op_name,
                    input_layouts=[get_layout(t) for t in tensor_args],
                    rule_name="fast_path",
                    plan=None,
                    pre_collectives=[],
                    redistributions=[],
                    post_collectives=[],
                    output_layout=None,
                    op_kwargs=dict(kwargs),
                )
            )
        return result

    # If runtime is not set but we have distributed tensors, fall back to raw op
    # This can happen during optimizer creation before training starts
    if runtime is None:
        raw = _get_raw(op_name)
        return raw(*args, **kwargs)

    input_layouts = [get_layout(t) for t in tensor_args]

    rule_fn = get_rule(op_name)
    if rule_fn is None:
        print(f"WARNING: No rule found for op {op_name}, falling back to replicated")
        return _fallback_replicated(op_name, tensor_args, other_args, kwargs)

    rule_kwargs = _rule_kwargs_for_op(op_name, other_args, kwargs)
    cache = runtime.plan_cache
    cache_key = (op_name, tuple(input_layouts), _hashable_kwargs(rule_kwargs))
    plan: Optional[ShardingPlan] = cache.get(cache_key)
    if plan is None:
        plan = rule_fn(*input_layouts, runtime=runtime, **rule_kwargs)
        cache.put(cache_key, plan)

    # Apply optional pre-collective per input (e.g. broadcast for input 0).
    # Rule sets pre_collectives[i] for each tensor input; None = no collective.
    pre_collectives_log: List[Dict[str, Any]] = []
    preprocessed = []
    layout_idx = 0
    for a in args:
        if hasattr(a, "get_value"):
            processed = a
            spec = (
                plan.pre_collectives[layout_idx]
                if plan.pre_collectives and layout_idx < len(plan.pre_collectives)
                else None
            )
            if isinstance(spec, CCL):
                processed = spec(processed)
                pre_collectives_log.append(spec.log_dict(arg_idx=layout_idx))
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
                redistributed.append(redistribute(a, target))
            else:
                redistributed.append(a)
            layout_idx += 1
        else:
            redistributed.append(a)

    raw = _get_raw(op_name)
    # Strip rule-only kwargs (e.g. gather_output) so raw C++ op is not passed them
    kwargs_for_raw = {k: v for k, v in kwargs.items() if k not in ("gather_output",)}
    # Non-tensor positionals are already interleaved into *redistributed* (same loop as
    # *preprocessed*); do not also splat *other_args* or scalars (e.g. epsilon) duplicate.
    result = raw(*redistributed, **kwargs_for_raw)

    # Apply optional post-collective per output. Rule sets post_collectives[j] for each output.
    # If result is a list/tuple of tensors, apply to each element; otherwise apply to the single output.
    post_collectives_log: List[Dict[str, Any]] = []
    if plan.post_collectives:
        is_multi = (
            isinstance(result, (list, tuple))
            and len(result) > 0
            and all(isinstance(x, ttml.autograd.Tensor) for x in result)
        )
        if is_multi:
            out_list = list(result)
            for i, out in enumerate(out_list):
                if i < len(plan.post_collectives):
                    spec = plan.post_collectives[i]
                    if isinstance(spec, CCL):
                        out_list[i] = spec(out)
                        post_collectives_log.append(spec.log_dict(out_idx=i))
            result = tuple(out_list) if isinstance(result, tuple) else out_list
        else:
            if len(plan.post_collectives) > 0:
                spec = plan.post_collectives[0]
                if isinstance(spec, CCL):
                    result = spec(result)
                    post_collectives_log.append(spec.log_dict(out_idx=0))

    if isinstance(result, ttml.autograd.Tensor):
        set_layout(result, plan.output_layout)

    if dispatch_trace.enabled:
        dispatch_trace.record(
            TraceEntry(
                op_name=op_name,
                input_layouts=input_layouts,
                rule_name=rule_fn.__name__ if rule_fn else None,
                plan=plan,
                pre_collectives=pre_collectives_log,
                redistributions=redistributions,
                post_collectives=post_collectives_log,
                output_layout=plan.output_layout,
                op_kwargs=dict(rule_kwargs),
            )
        )

    dispatch_trace.exit_dispatch()
    return result


# ---------------------------------------------------------------------------
# register_op decorator: wraps a raw callable into dispatch
# ---------------------------------------------------------------------------


def register_op(op_name: str, opp: Callable):
    """Register an op for distributed dispatch.

    Saves the raw callable and returns a wrapper that routes through
    ``dispatch(op_name, ...)``.

    This is the decorator that ``_register_ops`` uses at import-time to
    monkey-patch ``ttml.ops.*`` entry points.
    """
    _RAW_OPS[op_name] = opp

    @functools.wraps(opp)
    def wrapper(*args, **kwargs):
        return dispatch(op_name, *args, **kwargs)

    return wrapper
