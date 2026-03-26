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
    model = ttml.distributed.parallelize_module(
        model, mesh_device,
        {r".*\.w1": ColwiseParallel(), r".*\.w2": RowwiseParallel()},
        tp_axis=0,
    )
    # from here on, ttml.ops.linear.linear etc. go through dispatch
"""

from .layout import Layout, Shard, Replicate, get_layout, set_layout
from .mesh_runtime import MeshRuntime, get_runtime, set_runtime
from .dispatch import dispatch, register_op
from .redistribute import redistribute
from .cache import PlanCache
from .rules.registry import (
    ShardingPlan,
    CCL,
    Broadcast,
    AllReduce,
    AllGather,
    OptionalCCL,
    register_rule,
    get_rule,
    register_module_rule,
    get_module_rule,
)
from .debug import DispatchTracer, DispatchTraceCallback, dispatch_trace
from .training import distribute_tensor, parallelize_module
from .style import ParallelStyle, TpPlan, ColwiseParallel, RowwiseParallel
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
    "CCL",
    "Broadcast",
    "AllReduce",
    "AllGather",
    "OptionalCCL",
    "register_rule",
    "get_rule",
    "register_module_rule",
    "get_module_rule",
    "DispatchTracer",
    "DispatchTraceCallback",
    "dispatch_trace",
    "distribute_tensor",
    "parallelize_module",
    "sync_gradients",
    "ParallelStyle",
    "TpPlan",
    "ColwiseParallel",
    "RowwiseParallel",
    "init_ops",
]
