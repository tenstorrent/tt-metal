# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One DFlash drafter decoder layer: input_layernorm -> attention -> residual ->
post_attention_layernorm -> MLP -> residual. Mirrors Qwen3DFlashDecoderLayer (dflash.py)
exactly -- input_layernorm normalizes ONLY the noise/draft block, never the context (the
context was already normalized once, in compute_context, and re-read unchanged by every
layer's own k_proj/v_proj)."""

from __future__ import annotations

import ttnn
from models.demos.gemma4.tt.dflash.attention import dflash_attention_forward
from models.demos.gemma4.tt.dflash.mlp import dflash_mlp_forward
from models.demos.gemma4.tt.dflash.weights import DFlashLayerWeights


def dflash_layer_forward(
    context: ttnn.Tensor,
    noise: ttnn.Tensor,
    layer_weights: DFlashLayerWeights,
    cos_full: ttnn.Tensor,
    sin_full: ttnn.Tensor,
    attn_mask: ttnn.Tensor,
    mesh_config,
    ccl_manager,
    num_local_heads: int,
    num_local_kv_heads: int,
    head_dim: int,
    eps: float,
) -> ttnn.Tensor:
    residual = noise
    normed = layer_weights.input_layernorm(noise)
    attn_out = dflash_attention_forward(
        context,
        normed,
        layer_weights.attn,
        cos_full,
        sin_full,
        attn_mask,
        mesh_config,
        ccl_manager,
        num_local_heads,
        num_local_kv_heads,
        head_dim,
        eps,
    )
    noise = ttnn.add(residual, attn_out)

    residual = noise
    normed = layer_weights.post_attention_layernorm(noise)
    mlp_out = dflash_mlp_forward(normed, layer_weights.mlp, mesh_config=mesh_config, ccl_manager=ccl_manager)
    noise = ttnn.add(residual, mlp_out)
    return noise
