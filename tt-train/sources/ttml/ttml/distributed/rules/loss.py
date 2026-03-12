# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sharding rules for loss ops.

cross_entropy_loss needs the full (replicated) logits.  If the logits are
sharded (e.g. column-parallel classification head), the rule requests a
replicated input layout.  The dispatch layer uses ``all_gather`` with
``GradOutputType.REPLICATED`` so the backward divides by TP-size, giving
each device the correct gradient for its shard.
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
    """Logits must be replicated for loss computation.

    The target tensor (integer labels) is not distributed by TP.
    """
    if not layouts:
        return ShardingPlan(input_layouts=[], output_layout=replicated_layout(1))

    logit_layout = layouts[0]
    ndim = logit_layout.ndim
    rep = replicated_layout(ndim)

    input_layouts = [rep] + list(layouts[1:])
    return ShardingPlan(
        input_layouts=input_layouts,
        output_layout=rep,
        gather_grad_replicated=True,
    )
