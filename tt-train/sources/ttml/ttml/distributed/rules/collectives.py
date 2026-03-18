# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sharding rules for collective ops (broadcast, all_gather, all_reduce).

These ops are patched to go through dispatch so the tracer records them.
Rules are layout-only: output is replicated on the collective axis.
"""

from __future__ import annotations

from ..layout import Layout, Replicate
from .registry import ShardingPlan, register_rule


def _mesh_axis_from_kwargs(kwargs) -> int:
    return kwargs.get("cluster_axis", kwargs.get("mesh_axis", 0))


@register_rule("broadcast")
def broadcast_rule(
    input_layout: Layout,
    *,
    runtime=None,
    **kwargs,
) -> ShardingPlan:
    """Output is replicated on the broadcast axis."""
    mesh_axis = _mesh_axis_from_kwargs(kwargs)
    output_layout = input_layout.with_placement(mesh_axis, Replicate())
    return ShardingPlan(
        input_layouts=[input_layout],
        output_layout=output_layout,
    )


@register_rule("all_gather")
def all_gather_rule(
    input_layout: Layout,
    *,
    runtime=None,
    **kwargs,
) -> ShardingPlan:
    """Output is replicated on the gather axis (dim may change)."""
    mesh_axis = _mesh_axis_from_kwargs(kwargs)
    output_layout = input_layout.with_placement(mesh_axis, Replicate())
    return ShardingPlan(
        input_layouts=[input_layout],
        output_layout=output_layout,
    )


@register_rule("all_reduce")
def all_reduce_rule(
    input_layout: Layout,
    *,
    runtime=None,
    **kwargs,
) -> ShardingPlan:
    """Output is replicated on the reduce axis."""
    mesh_axis = _mesh_axis_from_kwargs(kwargs)
    output_layout = input_layout.with_placement(mesh_axis, Replicate())
    return ShardingPlan(
        input_layouts=[input_layout],
        output_layout=output_layout,
    )
