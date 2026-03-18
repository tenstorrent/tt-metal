# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Op and module rule registries.

Rules are registered with decorators and looked up at dispatch time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from ..layout import Layout


# ---------------------------------------------------------------------------
# Optional CCL types (per-input / per-output)
# ---------------------------------------------------------------------------


class CCL(ABC):
    """Base for optional collectives. Subclasses implement __call__ to run the CCL."""

    @abstractmethod
    def __call__(self, tensor: Any) -> Any:
        """Apply this collective to tensor and return the result."""
        ...

    def log_dict(
        self, *, arg_idx: Optional[int] = None, out_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """Return a dict for dispatch trace logging."""
        d: Dict[str, Any] = {}
        if arg_idx is not None:
            d["arg_idx"] = arg_idx
        if out_idx is not None:
            d["out_idx"] = out_idx
        return d


@dataclass(frozen=True)
class Broadcast(CCL):
    """Optional pre-collective: broadcast tensor on the given mesh axis."""

    mesh_axis: int

    def __call__(self, tensor: Any) -> Any:
        import ttml

        return ttml.ops.distributed.broadcast(tensor, cluster_axis=self.mesh_axis)

    def log_dict(
        self, *, arg_idx: Optional[int] = None, out_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        d = super().log_dict(arg_idx=arg_idx, out_idx=out_idx)
        d["type"] = "broadcast"
        d["mesh_axis"] = self.mesh_axis
        return d


@dataclass(frozen=True)
class AllReduce(CCL):
    """Optional post-collective: all_reduce on the given mesh axis."""

    mesh_axis: int
    noop_backward: bool = False

    def __call__(self, tensor: Any) -> Any:
        import ttml

        return ttml.ops.distributed.all_reduce(
            tensor,
            noop_backward=self.noop_backward,
            cluster_axis=self.mesh_axis,
        )

    def log_dict(
        self, *, arg_idx: Optional[int] = None, out_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        d = super().log_dict(arg_idx=arg_idx, out_idx=out_idx)
        d["type"] = "all_reduce"
        d["mesh_axis"] = self.mesh_axis
        d["noop_backward"] = self.noop_backward
        return d


@dataclass(frozen=True)
class AllGather(CCL):
    """Optional post-collective: all_gather (may change shape, e.g. LM head)."""

    dim: int
    mesh_axis: int
    gather_grad_replicated: bool = False

    def __call__(self, tensor: Any) -> Any:
        import ttml

        grad_type = (
            ttml.ops.distributed.GradOutputType.REPLICATED
            if self.gather_grad_replicated
            else ttml.ops.distributed.GradOutputType.SHARDED
        )
        return ttml.ops.distributed.all_gather(
            tensor,
            dim=self.dim,
            cluster_axis=self.mesh_axis,
            grad_output_type=grad_type,
        )

    def log_dict(
        self, *, arg_idx: Optional[int] = None, out_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        d = super().log_dict(arg_idx=arg_idx, out_idx=out_idx)
        d["type"] = "all_gather"
        d["dim"] = self.dim
        d["mesh_axis"] = self.mesh_axis
        d["grad_replicated"] = self.gather_grad_replicated
        return d


# None = no collective for that input/output
OptionalCCL = Optional[Union[Broadcast, AllReduce, AllGather]]


# ---------------------------------------------------------------------------
# Op sharding plans
# ---------------------------------------------------------------------------


@dataclass
class ShardingPlan:
    """The output of an op rule: tells dispatch how to handle one call.

    Pre- and post-collectives are optional CCLs. The op would run correctly
    without them; they are an opportunity to insert communication. The rule
    specifies them per input and per output using OptionalCCL (None | Broadcast
    | AllReduce | AllGather).

    Pre-collectives (per input):
        pre_collectives[i] is None or Broadcast(mesh_axis) for the i-th tensor input.

    Post-collectives (per output):
        post_collectives[j] is None, AllReduce(mesh_axis, noop_backward), or
        AllGather(dim, mesh_axis, gather_grad_replicated) for the j-th output.

    Attributes:
        input_layouts: Required layouts for each input tensor.
        output_layout: Layout of the output tensor (single output).
        pre_collectives: Optional list of OptionalCCL, one per tensor input.
        post_collectives: Optional list of OptionalCCL, one per output.
    """

    input_layouts: List[Layout]
    output_layout: Layout
    pre_collectives: Optional[List[OptionalCCL]] = None
    post_collectives: Optional[List[OptionalCCL]] = None


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
