# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Module-level transform rules for parallelize_module().

Each rule is registered with ``@register_module_rule(ModuleClass)`` and
receives:
    module      – the module instance to transform
    mesh_device – the mesh device to distribute tensors to
    For leaf modules (e.g. LinearLayer): policy and prefix for weight distribution.
    For composite modules (e.g. GroupedQueryAttention): tp_axis and cp_axis (no policy/prefix).

The rule must return the (possibly mutated) module.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional

from .layout import Layout, Shard, Replicate
from .rules.registry import register_module_rule

import ttml
from ttml.modules import LinearLayer, AbstractModuleBase

# ---------------------------------------------------------------------------
# GroupedQueryAttention
# ---------------------------------------------------------------------------


def _distribute_gqa(
    module,
    mesh_device,
    tp_axis: int,
    cp_axis: Optional[int] = None,
):
    """Distribute GQA: adjust local head/group counts and handle CP.

    tp_axis and cp_axis are passed through from parallelize_module (no policy parsing).
    Sub-linear weight sharding and forward collectives (broadcast, all_reduce) are
    applied when parallelize_module recurses into q_linear, kv_linear, out_linear
    and matches them to ColwiseParallel/RowwiseParallel. This rule only:
    - Adjusts num_heads and num_groups for TP
    - When cp_axis is provided: rebuilds rope_params and swaps to ring_attention_sdpa

    Reference: C++ DistributedGroupedQueryAttention in
    modules/distributed/grouped_query_attention.cpp
    """
    mesh_shape = mesh_device.shape

    # Handle TP: adjust local head/group counts (child linears get styles on recursion)
    if tp_axis is not None:
        tp_size = mesh_shape[tp_axis]
        num_heads = module.num_heads
        num_groups = module.num_groups

        if num_heads % tp_size != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by tp_size ({tp_size})"
            )
        if num_groups % tp_size != 0:
            raise ValueError(
                f"num_groups ({num_groups}) must be divisible by tp_size ({tp_size})"
            )

        module.num_heads = num_heads // tp_size
        module.num_groups = num_groups // tp_size

    # Handle CP
    if cp_axis is not None:
        cp_size = mesh_shape[cp_axis]
        if cp_size > 1:
            # Rebuild rope_params with CP sharding
            old_params = module.rope_params
            old_seq_len = old_params.sequence_length
            new_params = ttml.ops.rope.build_rope_params(
                old_seq_len,
                old_params.head_dim,
                old_params.theta,
                old_params.rope_scaling_params,
                cp_axis=cp_axis,
            )
            module.rope_params = new_params

            # Verify the update
            assert (
                module.rope_params.sequence_length == old_seq_len // cp_size
            ), f"rope_params not updated: expected {old_seq_len // cp_size}, got {module.rope_params.sequence_length}"

            # Swap to ring_attention_sdpa with cp_axis bound
            module.sdpa = partial(
                ttml.ops.distributed.ring_attention_sdpa, cp_axis=cp_axis
            )

    return module


from ttml.models.llama.gqattn import GroupedQueryAttention

register_module_rule(GroupedQueryAttention)(_distribute_gqa)
