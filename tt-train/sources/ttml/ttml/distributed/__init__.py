# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Layout-aware distributed dispatch for TTML.

Importing this package activates the dispatch layer: selected ttml.ops.*
entry points are replaced with dispatch-wrapped versions that automatically
handle tensor redistribution based on sharding rules.

Usage:
    import ttml
    import ttml.distributed  # activates dispatch

    mesh_device = ttml.autograd.AutoContext.get_instance().get_device()
    model = MyLlama(config)
    policy = {"layer.weight": Layout(placements=(Replicate(), Shard(-2)))}
    model = ttml.distributed.distribute_module(model, mesh_device, policy)
    # from here on, ttml.ops.linear.linear etc. go through dispatch
"""

from .layout import Layout, Shard, Replicate, get_layout, set_layout
from .mesh_runtime import MeshRuntime, get_runtime, set_runtime
from .dispatch import dispatch, register_op
from .redistribute import redistribute
from .cache import PlanCache
from .rules.registry import (
    ShardingPlan,
    register_rule,
    get_rule,
    register_module_rule,
    get_module_rule,
)
from .debug import DispatchTracer, dispatch_trace
from .training import distribute_module, distribute_tensor, sync_gradients
from ._register_ops import init_ops

from . import module_rules as _module_rules  # register module rules  # noqa: F401

__all__ = [
    "Layout",
    "Shard",
    "Replicate",
    "get_layout",
    "set_layout",
    "MeshRuntime",
    "get_runtime",
    "set_runtime",
    "dispatch",
    "register_op",
    "redistribute",
    "PlanCache",
    "ShardingPlan",
    "register_rule",
    "get_rule",
    "register_module_rule",
    "get_module_rule",
    "DispatchTracer",
    "dispatch_trace",
    "distribute_module",
    "distribute_tensor",
    "sync_gradients",
    "init_ops",
]
