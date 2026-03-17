# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Module-level transform rules for distribute_module().

Each rule is registered with ``@register_module_rule(ModuleClass)`` and
receives:
    module      – the module instance to transform
    mesh_device – the mesh device to distribute tensors to
    policy      – dict mapping param_name → Layout for weight distribution
    prefix      – parameter name prefix for policy lookup

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
# LinearLayer
# ---------------------------------------------------------------------------


def _infer_bias_layout(weight_layout: Layout) -> Layout:
    """Infer bias layout from weight layout.

    Column-parallel (weight sharded on dim -2): bias sharded on dim -1
    Row-parallel (weight sharded on dim -1): bias replicated
    """
    placements = []
    for p in weight_layout.placements:
        if isinstance(p, Shard) and p.dim in (-2, 2):
            placements.append(Shard(-1))
        else:
            placements.append(Replicate())
    return Layout(placements=tuple(placements))


@register_module_rule(LinearLayer)
def distribute_linear(
    module: LinearLayer,
    mesh_device,
    policy: Dict[str, Layout],
    prefix: str = "",
) -> LinearLayer:
    """Distribute a LinearLayer's weight (and optional bias) according to *policy*.

    Column-parallel: weight sharded on dim -2 (out_features), bias sharded on dim -1
    Row-parallel: weight sharded on dim -1 (in_features), bias stays replicated

    IMPORTANT: We must call override_tensor() to update the C++ side's m_named_tensors
    map. This ensures the old tensor can be deallocated and the optimizer (which calls
    model.parameters()) will get the new distributed tensors.
    """
    from .training import distribute_tensor, _match_policy

    weight_key = f"{prefix}.weight" if prefix else "weight"
    layout = _match_policy(weight_key, policy)
    if layout is not None:
        new_w = distribute_tensor(module.weight.tensor, mesh_device, layout)
        # Update both Python side (Parameter.tensor) and C++ side (m_named_tensors)
        module.weight.tensor = new_w
        module.override_tensor(new_w, "weight")

        if module.bias is not None:
            bias_key = f"{prefix}.bias" if prefix else "bias"
            bias_layout = _match_policy(bias_key, policy)
            if bias_layout is None:
                bias_layout = _infer_bias_layout(layout)
            new_b = distribute_tensor(module.bias.tensor, mesh_device, bias_layout)
            # Update both Python side and C++ side
            module.bias.tensor = new_b
            module.override_tensor(new_b, "bias")

    return module


# ---------------------------------------------------------------------------
# GroupedQueryAttention
# ---------------------------------------------------------------------------


def _distribute_gqa(
    module,
    mesh_device,
    policy: Dict[str, Layout],
    prefix: str = "",
    cp_axis: Optional[int] = None,
):
    """Distribute GQA: adjust local head/group counts, distribute sub-linears, and handle CP.

    The TP axis and size are inferred from the policy layouts. If no TP sharding
    is found in the policy for this module's weights, the module is returned
    unchanged (but CP handling still applies if cp_axis is provided).

    When cp_axis is provided:
    - Rebuilds rope_params with CP sharding
    - Swaps sdpa to ring_attention_sdpa with cp_axis bound

    Reference: C++ DistributedGroupedQueryAttention in
    modules/distributed/grouped_query_attention.cpp
    """
    from .training import distribute_tensor, _match_policy
    from ttml.models.llama.gqattn import GroupedQueryAttention

    mesh_shape = mesh_device.shape

    # Try to find TP axis and size from the policy
    q_weight_key = f"{prefix}.q_linear.weight" if prefix else "q_linear.weight"
    q_layout = _match_policy(q_weight_key, policy)

    # Handle TP if policy exists
    if q_layout is not None:
        # Find which axis has Shard placement to determine TP axis
        tp_axis = None
        for i, p in enumerate(q_layout.placements):
            if isinstance(p, Shard):
                tp_axis = i
                break

        if tp_axis is not None:
            tp_size = mesh_shape[tp_axis]

            if tp_size > 1:
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

                # Distribute sub-linear layers
                q_prefix = f"{prefix}.q_linear" if prefix else "q_linear"
                kv_prefix = f"{prefix}.kv_linear" if prefix else "kv_linear"
                out_prefix = f"{prefix}.out_linear" if prefix else "out_linear"

                distribute_linear(module.q_linear, mesh_device, policy, q_prefix)
                distribute_linear(module.kv_linear, mesh_device, policy, kv_prefix)
                distribute_linear(module.out_linear, mesh_device, policy, out_prefix)

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
