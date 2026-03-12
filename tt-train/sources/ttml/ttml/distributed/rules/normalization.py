# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sharding rules for normalization ops (layernorm, rmsnorm).

Normalization reduces over the last dimension.  If the input is sharded on
that dimension we need an all_gather first, but for TP (sharding on the
feature dim), the typical TTML layout is [B,1,S,H] where H is the hidden
dim and normalization operates over H.  In standard transformer TP the
hidden dim is *not* sharded for RMSNorm (it operates on the full hidden
dim), so the common path is pass-through.
"""

from __future__ import annotations

from ..layout import Layout, Shard, Replicate, replicated_layout
from .registry import ShardingPlan, register_rule


def _norm_rule(layouts, runtime=None, **kwargs):
    """Shared rule logic for layernorm and rmsnorm.

    ``layouts`` contains one Layout per tensor-typed argument (the scalar
    epsilon / optional beta are non-tensor and never appear here).
    """
    input_layout = layouts[0] if layouts else replicated_layout(1)
    return ShardingPlan(
        input_layouts=list(layouts),
        output_layout=input_layout,
    )


@register_rule("rmsnorm")
@register_rule("rmsnorm_composite")
def rmsnorm_rule(*layouts, runtime=None, **kwargs):
    return _norm_rule(layouts, runtime=runtime, **kwargs)


@register_rule("layernorm")
@register_rule("composite_layernorm")
def layernorm_rule(*layouts, runtime=None, **kwargs):
    return _norm_rule(layouts, runtime=runtime, **kwargs)
