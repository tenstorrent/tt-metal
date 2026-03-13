# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sharding rules for loss ops.

cross_entropy_loss needs the full logits along the vocab dimension (last dim).
If the logits are sharded on the last dimension (e.g. column-parallel output),
the rule gathers only that dimension while preserving other sharding (e.g. DP
batch sharding).
"""

from __future__ import annotations

from ..layout import Layout, Shard, Replicate, replicated_layout
from .registry import ShardingPlan, register_rule


@register_rule("cross_entropy_loss")
def cross_entropy_loss_rule(
    *layouts,
    runtime=None,
    **kwargs,
) -> ShardingPlan:
    """Gather logits along vocab dimension (last dim) for loss computation.

    Only gathers the last tensor dimension (vocab/logits). Other dimensions
    (e.g. batch sharded on DP axis) are preserved.
    """
    if not layouts:
        return ShardingPlan(input_layouts=[], output_layout=replicated_layout(1))

    logit_layout = layouts[0]

    # Build layout: gather any sharding on the last tensor dimension (dim 3 for 4D input),
    # keep all other placements as-is
    new_placements = []
    for p in logit_layout.placements:
        if isinstance(p, Shard) and p.dim in (-1, 3):
            # Shard on last dim (vocab) -> gather it
            new_placements.append(Replicate())
        else:
            # Keep other placements (preserves batch sharding, etc.)
            new_placements.append(p)
    logit_target_layout = Layout(placements=tuple(new_placements))

    # Target layout: keep as-is (preserves DP batch sharding)
    input_layouts = [logit_target_layout] + list(layouts[1:])
    return ShardingPlan(
        input_layouts=input_layouts,
        output_layout=logit_target_layout,
        gather_grad_replicated=True,
    )
