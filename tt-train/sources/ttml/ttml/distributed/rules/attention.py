# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sharding rules for attention, multi-head utils, RoPE, embedding, and reshape.

SDPA: Q/K/V are sharded on the heads dimension (dim 0 in [H, B, S, D]),
      which corresponds to TP sharding across devices.  Each device runs
      local attention on its own head shard; no cross-device communication.

Multi-head utils: grouped_heads_creation, heads_fusion, heads_creation pass
through whatever layout the input has.  The op internally reshapes but the
sharding axis is preserved.

RoPE: Layout passes through.
Embedding: Layout passes through.
Reshape: Layout passes through (conservative; specific reshapes may need rules).
"""

from __future__ import annotations

from ..layout import Layout, Replicate, Shard
from .registry import ShardingPlan, register_rule


# -- SDPA --------------------------------------------------------------------


@register_rule("sdpa")
def sdpa_rule(
    *layouts,
    runtime=None,
    **kwargs,
) -> ShardingPlan:
    q_layout = layouts[0] if layouts else Layout((Replicate(),))
    return ShardingPlan(
        input_layouts=list(layouts),
        output_layout=q_layout,
    )


# ring_sdpa uses the same sharding rule as sdpa (trace shows "ring_sdpa" when ring_attention_sdpa is used)
register_rule("ring_sdpa")(sdpa_rule)


# -- Multi-head utils --------------------------------------------------------


@register_rule("grouped_heads_creation")
def grouped_heads_creation_rule(
    *layouts,
    runtime=None,
    **kwargs,
) -> ShardingPlan:
    q_layout = layouts[0] if layouts else Layout((Replicate(),))
    return ShardingPlan(
        input_layouts=list(layouts),
        output_layout=q_layout,
    )


@register_rule("heads_fusion")
def heads_fusion_rule(
    *layouts,
    runtime=None,
    **kwargs,
) -> ShardingPlan:
    input_layout = layouts[0] if layouts else Layout((Replicate(),))
    return ShardingPlan(
        input_layouts=list(layouts),
        output_layout=input_layout,
    )


@register_rule("heads_creation")
def heads_creation_rule(
    *layouts,
    runtime=None,
    **kwargs,
) -> ShardingPlan:
    input_layout = layouts[0] if layouts else Layout((Replicate(),))
    return ShardingPlan(
        input_layouts=list(layouts),
        output_layout=input_layout,
    )


# -- RoPE -------------------------------------------------------------------


@register_rule("rope")
def rope_rule(
    *layouts,
    runtime=None,
    **kwargs,
) -> ShardingPlan:
    input_layout = layouts[0] if layouts else Layout((Replicate(),))
    return ShardingPlan(
        input_layouts=list(layouts),
        output_layout=input_layout,
    )


# -- Embedding ---------------------------------------------------------------


def _embedding_output_layout(input_layout: Layout) -> Layout:
    """Remap input layout to embedding output layout.

    Embedding input is (B, 1, 1, S) so sequence is dim 3; output is (B, 1, S, D)
    so sequence is dim 2. Preserve sequence sharding by remapping Shard(3) -> Shard(2).
    Shard(-1) on input (last dim = S) -> Shard(2) on output (seq dim).
    """
    new_placements = []
    for p in input_layout.placements:
        if isinstance(p, Shard):
            if p.dim in (3, -1):
                new_placements.append(Shard(2))
            else:
                new_placements.append(p)
        else:
            new_placements.append(p)
    return Layout(placements=tuple(new_placements))


@register_rule("embedding")
def embedding_rule(
    *layouts,
    runtime=None,
    **kwargs,
) -> ShardingPlan:
    input_layout = layouts[0] if layouts else Layout((Replicate(),))
    output_layout = _embedding_output_layout(input_layout)
    return ShardingPlan(
        input_layouts=list(layouts),
        output_layout=output_layout,
    )


# -- Reshape -----------------------------------------------------------------


@register_rule("reshape")
def reshape_rule(
    *layouts,
    runtime=None,
    **kwargs,
) -> ShardingPlan:
    input_layout = layouts[0] if layouts else Layout((Replicate(),))
    return ShardingPlan(
        input_layouts=list(layouts),
        output_layout=input_layout,
    )
