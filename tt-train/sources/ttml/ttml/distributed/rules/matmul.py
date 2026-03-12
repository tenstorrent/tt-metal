# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sharding rules for linear and matmul ops.

Column-parallel:
    weight sharded on out_features (dim -2 in TTML's [1,1,out,in] layout)
    input replicated → output sharded same as weight → no post-collective

Row-parallel:
    weight sharded on in_features (dim -1 in [1,1,out,in] layout)
    input sharded on last dim → output is partial sum → all_reduce
"""

from __future__ import annotations

from ..layout import Layout, Shard, Replicate, replicated_layout
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


@register_rule("linear")
def linear_rule(
    input_layout: Layout,
    weight_layout: Layout,
    *extra_layouts,
    runtime=None,
    **kwargs,
) -> ShardingPlan:
    """Linear op: y = x @ W^T (+ bias).

    Weight shape is [1, 1, out_features, in_features].
    Column-parallel shards dim -2 (out_features), row-parallel shards dim -1 (in_features).

    Column-parallel:
        - Pre: broadcast input (no-op forward, all_reduce backward)
        - Post: none (output is sharded)

    Row-parallel:
        - Pre: scatter input if not already sharded
        - Post: all_reduce with noop_backward=True if input was sharded
          (to avoid double all_reduce from column-parallel's broadcast backward)
    """
    ndim = weight_layout.ndim
    rep = replicated_layout(ndim)

    # Column parallel: weight sharded on dim -2 (out_features)
    if _is_shard_on(weight_layout, -2) or _is_shard_on(weight_layout, 2):
        shard_dim = -2 if _is_shard_on(weight_layout, -2) else 2
        tp_axis = _find_shard_axis(weight_layout, shard_dim)

        bias_layouts = []
        for bl in extra_layouts:
            if bl is not None:
                bias_layouts.append(rep.with_placement(tp_axis, Shard(-1)))
            else:
                bias_layouts.append(None)

        # Output is sharded on last dim (out_features after matmul)
        output_layout = rep.with_placement(tp_axis, Shard(-1))
        return ShardingPlan(
            input_layouts=[rep, weight_layout] + bias_layouts,
            output_layout=output_layout,
            # Broadcast input to ensure all TP devices have same data
            # broadcast: no-op forward, all_reduce backward
            pre_collective="broadcast",
            pre_collective_mesh_axis=tp_axis,
        )

    # Row parallel: weight sharded on dim -1 (in_features)
    if _is_shard_on(weight_layout, -1) or _is_shard_on(weight_layout, 3):
        shard_dim = -1 if _is_shard_on(weight_layout, -1) else 3
        tp_axis = _find_shard_axis(weight_layout, shard_dim)

        # Check if input is already sharded (coming from column-parallel)
        input_is_sharded = _is_shard_on(input_layout, -1)

        # Input must be sharded on last dim to match weight's in_features sharding
        required_input = rep.with_placement(tp_axis, Shard(-1))
        bias_layouts = [None] * len(extra_layouts)
        return ShardingPlan(
            input_layouts=[required_input, weight_layout] + bias_layouts,
            output_layout=rep,
            post_collective="all_reduce",
            reduce_mesh_axis=tp_axis,
            # If input was already sharded (from column-parallel), use noop_backward
            # to avoid double all_reduce (column-parallel's broadcast does all_reduce in backward)
            noop_backward=input_is_sharded,
        )

    return ShardingPlan(
        input_layouts=[rep, rep]
        + [rep if bl is not None else None for bl in extra_layouts],
        output_layout=rep,
    )


@register_rule("matmul")
def matmul_rule(
    a_layout: Layout,
    b_layout: Layout,
    *extra_layouts,
    runtime=None,
    **kwargs,
) -> ShardingPlan:
    ndim = max(a_layout.ndim, b_layout.ndim)
    rep = replicated_layout(ndim)

    if _is_shard_on(b_layout, -1):
        tp_axis = _find_shard_axis(b_layout, -1)
        output_layout = rep.with_placement(tp_axis, Shard(-1))
        return ShardingPlan(
            input_layouts=[rep, b_layout],
            output_layout=output_layout,
        )

    if _is_shard_on(b_layout, -2):
        tp_axis = _find_shard_axis(b_layout, -2)
        required_a = a_layout.with_placement(tp_axis, Shard(-1))
        return ShardingPlan(
            input_layouts=[required_a, b_layout],
            output_layout=rep,
            post_collective="all_reduce",
            reduce_mesh_axis=tp_axis,
        )

    return ShardingPlan(input_layouts=[rep, rep], output_layout=rep)
