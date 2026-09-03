# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chain all 5 DFlash drafter layers. Context and RoPE tables are fixed across layers;
only the noise/draft-block representation threads from one layer to the next. Each
layer's mask depends only on (is_causal, sliding_window) -- built once per distinct
combination and reused (models/demos/gemma4/docs/dflash_design.md section 1: layers 1-4
are sliding/causal, layer 5 is full/bidirectional)."""

from __future__ import annotations

import ttnn
from models.demos.gemma4.tt.dflash.attention import build_attention_mask_additive
from models.demos.gemma4.tt.dflash.layer import dflash_layer_forward
from models.demos.gemma4.tt.dflash.weights import Gemma4DFlashWeights


def dflash_drafter_forward(
    context: ttnn.Tensor,
    noise: ttnn.Tensor,
    weights: Gemma4DFlashWeights,
    cos_full: ttnn.Tensor,
    sin_full: ttnn.Tensor,
    mesh_device,
    mesh_config,
    ccl_manager,
    num_local_heads: int,
    num_local_kv_heads: int,
    head_dim: int,
    eps: float,
    layer_configs: list[tuple[bool, int | None]],  # (is_causal, sliding_window) per layer
) -> ttnn.Tensor:
    ctx_len = context.shape[-2]
    q_len = noise.shape[-2]

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    mask_cache: dict[tuple[bool, int | None], ttnn.Tensor] = {}

    def mask_for(is_causal, sliding_window):
        key = (is_causal, sliding_window)
        if key not in mask_cache:
            mask_torch = build_attention_mask_additive(ctx_len, q_len, is_causal, sliding_window)
            mask_cache[key] = ttnn.from_torch(
                mask_torch, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
            )
        return mask_cache[key]

    for layer_weights, (is_causal, sliding_window) in zip(weights.layers, layer_configs):
        mask = mask_for(is_causal, sliding_window)
        noise = dflash_layer_forward(
            context,
            noise,
            layer_weights,
            cos_full,
            sin_full,
            mask,
            mesh_config,
            ccl_manager,
            num_local_heads,
            num_local_kv_heads,
            head_dim,
            eps,
        )
    return noise
