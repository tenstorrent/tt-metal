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
from .mesh_runtime import get_runtime


def _redistribute_impl(
    tensor, current_layout: Layout, target_layout: Layout, grad_replicated: bool = False
):
    """Perform redistribution using ttml autograd-aware collectives.

    These ops (all_gather, scatter) are autograd-aware - they build the
    backward graph automatically in C++. No Python backward needed.
    """
    result = tensor
    for mesh_axis, (cur, tgt) in enumerate(
        zip(current_layout.placements, target_layout.placements)
    ):
        if cur == tgt:
            continue

        if isinstance(cur, Shard) and isinstance(tgt, Replicate):
            # Shard -> Replicate: all_gather
            ag_kwargs = dict(dim=cur.dim, cluster_axis=mesh_axis)
            if grad_replicated:
                ag_kwargs[
                    "grad_output_type"
                ] = ttml.ops.distributed.GradOutputType.REPLICATED
            result = ttml.ops.distributed.all_gather(result, **ag_kwargs)
        elif isinstance(cur, Replicate) and isinstance(tgt, Shard):
            # Replicate -> Shard: scatter
            result = ttml.ops.distributed.scatter(
                result, dim=tgt.dim, cluster_axis=mesh_axis
            )
        elif isinstance(cur, Shard) and isinstance(tgt, Shard):
            # Shard(dim_a) -> Shard(dim_b): all_gather then scatter
            ag_kwargs = dict(dim=cur.dim, cluster_axis=mesh_axis)
            if grad_replicated:
                ag_kwargs[
                    "grad_output_type"
                ] = ttml.ops.distributed.GradOutputType.REPLICATED
            result = ttml.ops.distributed.all_gather(result, **ag_kwargs)
            result = ttml.ops.distributed.scatter(
                result, dim=tgt.dim, cluster_axis=mesh_axis
            )

    set_layout(result, target_layout)
    return result


def redistribute(tensor, target_layout: Layout, grad_replicated: bool = False):
    """Transform tensor to *target_layout*.  Differentiable for autograd.

    No-op if the tensor already has the requested layout.

    When *grad_replicated* is True, ``all_gather`` uses
    ``GradOutputType.REPLICATED`` (backward divides by TP-size).

    Uses ttml.ops.distributed collectives which are autograd-aware - backward
    is handled automatically by the C++ autograd graph.
    """
    current_layout = get_layout(tensor)
    if current_layout is None:
        set_layout(tensor, target_layout)
        return tensor
    if current_layout == target_layout:
        return tensor
    return _redistribute_impl(tensor, current_layout, target_layout, grad_replicated)
