# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sharding rules for elementwise ops.

Binary ops: inputs must match layouts.  We pick the "more sharded" one.
Unary ops: layout passes through unchanged.
"""

from __future__ import annotations

from ..layout import Layout, Shard, Replicate
from .registry import ShardingPlan, register_rule


def _shard_count(layout: Layout) -> int:
    return sum(1 for p in layout.placements if isinstance(p, Shard))


def _pick_target(a: Layout, b: Layout) -> Layout:
    """Choose the more-sharded layout as the target."""
    if _shard_count(a) >= _shard_count(b):
        return a
    return b


# -- Binary ------------------------------------------------------------------


@register_rule("add")
@register_rule("sub")
@register_rule("mul")
@register_rule("div")
def elementwise_binary_rule(
    a_layout: Layout,
    b_layout: Layout = None,
    *extra,
    runtime=None,
    **kwargs,
) -> ShardingPlan:
    if b_layout is None:
        return ShardingPlan(
            input_layouts=[a_layout],
            output_layout=a_layout,
        )
    target = _pick_target(a_layout, b_layout)
    return ShardingPlan(
        input_layouts=[target, target],
        output_layout=target,
    )


# -- Unary -------------------------------------------------------------------


@register_rule("relu")
@register_rule("gelu")
@register_rule("silu")
def elementwise_unary_rule(
    *layouts,
    runtime=None,
    **kwargs,
) -> ShardingPlan:
    input_layout = layouts[0] if layouts else Layout((Replicate(),))
    return ShardingPlan(
        input_layouts=list(layouts),
        output_layout=input_layout,
    )


# -- Dropout (behaves like unary, preserves layout) --------------------------


@register_rule("dropout")
def dropout_rule(
    *layouts,
    runtime=None,
    **kwargs,
) -> ShardingPlan:
    input_layout = layouts[0] if layouts else Layout((Replicate(),))
    return ShardingPlan(
        input_layouts=list(layouts),
        output_layout=input_layout,
    )
