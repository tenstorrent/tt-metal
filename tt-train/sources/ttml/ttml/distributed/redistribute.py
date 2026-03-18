# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Differentiable redistribute: transforms tensor layouts across mesh axes.

Uses ttml.ops.distributed collectives which are autograd-aware - they build
the backward graph automatically in C++. No Python Function wrapper needed.
"""

from __future__ import annotations

from typing import Optional

import ttml

from .layout import Layout, Shard, Replicate, get_layout, set_layout


def _redistribute_impl(tensor, current_layout: Layout, target_layout: Layout):
    """Perform redistribution using ttml autograd-aware collectives.

    Uses regular all_gather / scatter. Gradient handling (e.g. GradOutputType.REPLICATED)
    is a post_collective concern and is passed as post_collective args in dispatch.
    """
    result = tensor
    for mesh_axis, (cur, tgt) in enumerate(
        zip(current_layout.placements, target_layout.placements)
    ):
        if cur == tgt:
            continue

        if isinstance(cur, Shard) and isinstance(tgt, Replicate):
            # Shard -> Replicate: all_gather
            result = ttml.ops.distributed.all_gather(
                result, dim=cur.dim, cluster_axis=mesh_axis
            )
        elif isinstance(cur, Replicate) and isinstance(tgt, Shard):
            # Replicate -> Shard: scatter
            result = ttml.ops.distributed.scatter(
                result, dim=tgt.dim, cluster_axis=mesh_axis
            )
        elif isinstance(cur, Shard) and isinstance(tgt, Shard):
            # Shard(dim_a) -> Shard(dim_b): all_gather then scatter
            result = ttml.ops.distributed.all_gather(
                result, dim=cur.dim, cluster_axis=mesh_axis
            )
            result = ttml.ops.distributed.scatter(
                result, dim=tgt.dim, cluster_axis=mesh_axis
            )

    set_layout(result, target_layout)
    return result


def redistribute(tensor, target_layout: Layout):
    """Transform tensor to *target_layout*.  Differentiable for autograd.

    No-op if the tensor already has the requested layout. Uses regular all_gather/scatter.
    Gradient handling for collectives (e.g. grad_output_type) is configured via
    post_collective args in the dispatch layer, not here.
    """
    current_layout = get_layout(tensor)
    if current_layout is None:
        set_layout(tensor, target_layout)
        return tensor
    if current_layout == target_layout:
        return tensor
    return _redistribute_impl(tensor, current_layout, target_layout)
