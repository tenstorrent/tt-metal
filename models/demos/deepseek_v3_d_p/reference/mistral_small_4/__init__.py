# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Small-4-119B CPU reference helpers (block-level composition over MLAReference + torch MoE).

Follows the GLM-5.1 pattern: there is no runnable HF ``reference_model_cls`` for this model, so the
tests call ``mistral4_decoder_layer_reference`` directly and it composes the references this repo
already owns. ``rms_norm`` / ``dense_ffn`` are re-exported from ``glm_5_1.block`` so a caller needs
only this package.
"""

from models.demos.deepseek_v3_d_p.reference.glm_5_1.block import dense_ffn, rms_norm
from models.demos.deepseek_v3_d_p.reference.mistral_small_4.block import (
    llama4_attn_scale,
    llama4_attn_scale_params,
    mistral4_decoder_layer_reference,
)
from models.demos.deepseek_v3_d_p.reference.mistral_small_4.moe import (
    mistral4_moe_reference,
    mistral4_route_tokens_to_experts,
    mistral4_router_logits,
    mistral4_torch_config,
    unpack_stacked_expert_weights,
)

__all__ = [
    "dense_ffn",
    "llama4_attn_scale",
    "llama4_attn_scale_params",
    "mistral4_decoder_layer_reference",
    "mistral4_moe_reference",
    "mistral4_route_tokens_to_experts",
    "mistral4_router_logits",
    "mistral4_torch_config",
    "rms_norm",
    "unpack_stacked_expert_weights",
]
