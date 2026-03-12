# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Op and module rule registries.

Rules are registered with decorators and looked up at dispatch time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from ..layout import Layout


# ---------------------------------------------------------------------------
# Op sharding plans
# ---------------------------------------------------------------------------


@dataclass
class ShardingPlan:
    """The output of an op rule: tells dispatch how to handle one call.

    Attributes:
        input_layouts: Required layouts for each input tensor.
        output_layout: Layout of the output tensor.
        pre_collective: Collective to apply to input before op ("broadcast", etc).
        pre_collective_mesh_axis: Mesh axis for pre_collective.
        post_collective: Collective to apply after op ("all_reduce", "all_gather").
        reduce_mesh_axis: Mesh axis for post_collective.
        noop_backward: If True, post_collective has no-op backward (avoids double all_reduce).
        gather_grad_replicated: If True, all_gather backward divides by TP size.
    """

    input_layouts: List[Layout]
    output_layout: Layout
    pre_collective: Optional[str] = None
    pre_collective_mesh_axis: Optional[int] = None
    post_collective: Optional[str] = None
    reduce_mesh_axis: Optional[int] = None
    noop_backward: bool = False
    gather_grad_replicated: bool = False


# ---------------------------------------------------------------------------
# Op rule registry
# ---------------------------------------------------------------------------

_OP_RULES: Dict[str, Callable[..., ShardingPlan]] = {}


def register_rule(op_name: str):
    """Decorator: register a sharding rule for *op_name*.

    Usage::

        @register_rule("linear")
        def linear_rule(input_layout, weight_layout, *, runtime, **kw):
            ...
            return ShardingPlan(...)
    """

    def decorator(fn: Callable[..., ShardingPlan]):
        _OP_RULES[op_name] = fn
        return fn

    return decorator


def get_rule(op_name: str) -> Optional[Callable[..., ShardingPlan]]:
    return _OP_RULES.get(op_name)


# ---------------------------------------------------------------------------
# Module rule registry
# ---------------------------------------------------------------------------

_MODULE_RULES: Dict[Any, Callable] = {}


def register_module_rule(module_type):
    """Decorator: register a module transform rule.

    The key can be a class or a string path.

    Usage::

        @register_module_rule(LinearLayer)
        def distribute_linear(module, mesh_runtime, policy):
            ...
            return transformed_module
    """

    def decorator(fn: Callable):
        _MODULE_RULES[module_type] = fn
        return fn

    return decorator


def get_module_rule(module_type) -> Optional[Callable]:
    rule = _MODULE_RULES.get(module_type)
    if rule is not None:
        return rule
    for registered_type, fn in _MODULE_RULES.items():
        if isinstance(registered_type, type) and isinstance(module_type, type):
            if issubclass(module_type, registered_type):
                return fn
        elif isinstance(registered_type, type) and not isinstance(module_type, type):
            if isinstance(module_type, registered_type):
                return fn
    return None
