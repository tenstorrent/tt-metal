# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Layout-aware distributed dispatch for TTML.

Importing this package activates the dispatch layer: selected ttml.ops.*
entry points are replaced with dispatch-wrapped versions that automatically
handle tensor redistribution based on sharding rules.

Usage:
    model = Llama(config, mesh_device=mesh, tp_plan=ParallelizationPlan({...}, tp_axis=1))
    # TransformerBase materializes weights and parallelizes automatically
"""

from .layout import DistributedLayout, Shard, Replicate, get_layout, set_layout

# Backward-compatibility alias
Layout = DistributedLayout
from .mesh_runtime import MeshRuntime, get_runtime, set_runtime
from .dispatch import dispatch, register_op
from .redistribute import redistribute
from .rules.registry import (
    ShardingPlan,
    CCL,
    Broadcast,
    AllReduce,
    AllGather,
    OptionalCCL,
    register_rule,
    get_rule,
)
from .debug import DispatchTracer, DispatchTraceCallback, dispatch_trace
from .style import ParallelStyle, ColwiseParallel, RowwiseParallel
from ._register_ops import init_ops

__all__ = [
    "DistributedLayout",
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
    "ShardingPlan",
    "CCL",
    "Broadcast",
    "AllReduce",
    "AllGather",
    "OptionalCCL",
    "register_rule",
    "get_rule",
    "DispatchTracer",
    "DispatchTraceCallback",
    "dispatch_trace",
    "ParallelStyle",
    "ColwiseParallel",
    "RowwiseParallel",
    "init_ops",
]
