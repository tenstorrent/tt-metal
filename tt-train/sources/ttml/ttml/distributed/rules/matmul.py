# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sharding rules for linear and matmul ops.

Layout-only rules: styles (ColwiseParallel / RowwiseParallel) wrap the module
with pre/post collectives (broadcast, all_gather, all_reduce). This rule just
specifies input/output layouts for dispatch to redistribute and stamp.

Column-parallel: weight sharded on -2, input replicated, output sharded.
Row-parallel: weight sharded on -1, input sharded, output replicated.
"""

from __future__ import annotations

from ..layout import Layout, Shard, Replicate
from .registry import ShardingPlan, register_rule


def _is_shard_on(layout: Layout, dim: int) -> bool:
    for p in layout.placements:
        if isinstance(p, Shard) and p.dim == dim:
            return True
    return False


def _find_shard_axis(layout: Layout, dim: int) -> int:
    for axis, p in enumerate(layout.placements):
        if isinstance(p, Shard) and p.dim == dim:
            return axis
    return -1


def _linear_matmul_plan(
    input_layout: Layout,
    weight_layout: Layout,
    *extra_layouts,
    runtime=None,
    **kwargs,
) -> ShardingPlan:
    """Single plan for linear and matmul. Styles handle pre/post collectives."""
    # Column parallel: weight sharded on dim -2 (out_features)
    if _is_shard_on(weight_layout, -2) or _is_shard_on(weight_layout, 2):
        shard_dim = -2 if _is_shard_on(weight_layout, -2) else 2
        tp_axis = _find_shard_axis(weight_layout, shard_dim)
        output_layout = input_layout.with_placement(tp_axis, Shard(-1))
        bias_layouts = []
        for bl in extra_layouts:
            if bl is not None:
                bias_layouts.append(input_layout.with_placement(tp_axis, Shard(-1)))
            else:
                bias_layouts.append(None)
        return ShardingPlan(
            input_layouts=[input_layout, weight_layout] + bias_layouts,
            output_layout=output_layout,
        )

    # Row parallel: weight sharded on dim -1 (in_features)
    if _is_shard_on(weight_layout, -1) or _is_shard_on(weight_layout, 3):
        shard_dim = -1 if _is_shard_on(weight_layout, -1) else 3
        tp_axis = _find_shard_axis(weight_layout, shard_dim)
        required_input = input_layout.with_placement(tp_axis, Shard(-1))
        output_layout = input_layout.with_placement(tp_axis, Replicate())
        bias_layouts = [None] * len(extra_layouts)
        return ShardingPlan(
            input_layouts=[required_input, weight_layout] + bias_layouts,
            output_layout=output_layout,
        )

    # Fallback
    return ShardingPlan(
        input_layouts=[input_layout, weight_layout] + [bl for bl in extra_layouts],
        output_layout=input_layout,
    )


# Same rule for both ops; styles (Colwise/RowwiseParallel) add pre/post collectives
register_rule("linear")(_linear_matmul_plan)
register_rule("matmul")(_linear_matmul_plan)
