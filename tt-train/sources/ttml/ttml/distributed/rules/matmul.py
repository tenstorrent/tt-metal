# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sharding rules for linear and matmul ops.

Column-parallel:
    weight sharded on out_features (dim -2 in TTML's [1,1,out,in] layout)
    input replicated on TP axis → output sharded same as weight → no post-collective

Row-parallel:
    weight sharded on in_features (dim -1 in [1,1,out,in] layout)
    input sharded on last dim → output is partial sum → all_reduce

Rules only modify the specific mesh axis needed for TP. All other axes
(e.g., DP batch sharding) are left unchanged from the input layout.
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
        - Pre: broadcast input on TP axis (no-op forward, all_reduce backward)
        - Post: none (output is sharded on TP axis)

    Row-parallel:
        - Pre: scatter input if not already sharded
        - Post: all_reduce with noop_backward=True if input was sharded
          (to avoid double all_reduce from column-parallel's broadcast backward)

    Only the TP axis placement is modified. All other axes are left unchanged.
    """
    # Column parallel: weight sharded on dim -2 (out_features)
    if _is_shard_on(weight_layout, -2) or _is_shard_on(weight_layout, 2):
        shard_dim = -2 if _is_shard_on(weight_layout, -2) else 2
        tp_axis = _find_shard_axis(weight_layout, shard_dim)

        # Input: keep as-is (only broadcast on TP axis in pre_collective)
        # Output: same as input but with Shard(-1) on TP axis
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
            # Broadcast input on TP axis to ensure all TP devices have same data
            # broadcast: no-op forward, all_reduce backward
            pre_collective="broadcast",
            pre_collective_mesh_axis=tp_axis,
        )

    # Row parallel: weight sharded on dim -1 (in_features)
    if _is_shard_on(weight_layout, -1) or _is_shard_on(weight_layout, 3):
        shard_dim = -1 if _is_shard_on(weight_layout, -1) else 3
        tp_axis = _find_shard_axis(weight_layout, shard_dim)

        # Check if input is already sharded on last dim (coming from column-parallel)
        input_is_sharded = _is_shard_on(input_layout, -1)

        # Input must be sharded on last dim on TP axis
        required_input = input_layout.with_placement(tp_axis, Shard(-1))

        # Output: same as input but replicated on TP axis (after all_reduce)
        output_layout = input_layout.with_placement(tp_axis, Replicate())

        bias_layouts = [None] * len(extra_layouts)
        return ShardingPlan(
            input_layouts=[required_input, weight_layout] + bias_layouts,
            output_layout=output_layout,
            post_collective="all_reduce",
            reduce_mesh_axis=tp_axis,
            # If input was already sharded (from column-parallel), use noop_backward
            # to avoid double all_reduce (column-parallel's broadcast does all_reduce in backward)
            noop_backward=input_is_sharded,
        )

    # Fallback: no TP sharding, pass through input layout
    return ShardingPlan(
        input_layouts=[input_layout, weight_layout] + [bl for bl in extra_layouts],
        output_layout=input_layout,
    )


@register_rule("matmul")
def matmul_rule(
    a_layout: Layout,
    b_layout: Layout,
    *extra_layouts,
    runtime=None,
    **kwargs,
) -> ShardingPlan:
    """Matmul rule: only modify TP axis, leave all other axes unchanged."""
    if _is_shard_on(b_layout, -1):
        tp_axis = _find_shard_axis(b_layout, -1)
        # Output: same as input A but with Shard(-1) on TP axis
        output_layout = a_layout.with_placement(tp_axis, Shard(-1))
        return ShardingPlan(
            input_layouts=[a_layout, b_layout],
            output_layout=output_layout,
        )

    if _is_shard_on(b_layout, -2):
        tp_axis = _find_shard_axis(b_layout, -2)
        # Input A must be sharded on last dim on TP axis
        required_a = a_layout.with_placement(tp_axis, Shard(-1))
        # Output: same as input A but replicated on TP axis
        output_layout = a_layout.with_placement(tp_axis, Replicate())
        return ShardingPlan(
            input_layouts=[required_a, b_layout],
            output_layout=output_layout,
            post_collective="all_reduce",
            reduce_mesh_axis=tp_axis,
        )

    # Fallback: pass through input layout
    return ShardingPlan(input_layouts=[a_layout, b_layout], output_layout=a_layout)
