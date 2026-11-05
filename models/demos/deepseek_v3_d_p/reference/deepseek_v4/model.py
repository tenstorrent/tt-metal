# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A DeepSeek-V4 layer stack on CPU, composed from the reference modules, plus its weights in the layout
TtV4Transformer wants.

The same dataflow as ``DeepseekV4Model.forward``: embed, expand to ``hc_mult`` streams, the decoder
layers, ``hc_head`` to collapse, then the final RMSNorm. The layers are ``reference.deepseek_v4.block``'s,
so the stack and the block test share one per-layer reference.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.block import (
    build_v4_block_reference,
    v4_block_forward,
    v4_block_state_dict,
    v4_mhc_weights,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4HyperHead,
    DeepseekV4RMSNorm,
)


def build_v4_model_reference(config, seed: int = 0) -> dict:
    """A randomised V4 stack of ``config.num_hidden_layers`` layers.

    Each layer is ``build_v4_block_reference(config, i, seed + i)`` with a bf16 MoE: in fp32 a Pro
    layer's experts alone are ~100 GB of host memory. The head gets what the model prescribes
    (``DeepseekV4PreTrainedModel._init_weights``: fn ~ N(0, initializer_range), base zero, scale one)
    and the final norm a gain away from 1.0, for the reason the block's norms do.
    """
    layers = [
        build_v4_block_reference(config, i, seed=seed + i, moe_dtype=torch.bfloat16)
        for i in range(config.num_hidden_layers)
    ]
    torch.manual_seed(seed + config.num_hidden_layers)
    embed = nn.Embedding(config.vocab_size, config.hidden_size).eval()
    hc_head = DeepseekV4HyperHead(config).eval()
    norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps).eval()
    with torch.no_grad():
        hc_head.hc_fn.normal_(0.0, config.initializer_range)
        hc_head.hc_base.zero_()
        hc_head.hc_scale.fill_(1.0)
        norm.weight.uniform_(0.5, 1.5)
    return {"embed": embed, "layers": layers, "hc_head": hc_head, "norm": norm}


def v4_embed_streams(ref: dict, config, input_ids: torch.Tensor) -> torch.Tensor:
    """``input_ids`` [1, seq] -> the expanded residual [1, seq, hc_mult, hidden] that layer 0 is fed."""
    with torch.no_grad():
        return ref["embed"](input_ids).unsqueeze(2).expand(-1, -1, config.hc_mult, -1).contiguous()


def v4_model_forward(ref: dict, config, input_ids: torch.Tensor, first_layer_idx: int = 0, hidden=None):
    """One unchunked pass over layers ``[first_layer_idx, num_hidden_layers)``.

    ``hidden`` replaces the embedding as the input to ``first_layer_idx`` when given, which is how a
    later pipeline rank's slice is scored. Returns ``(norm(hc_head(h)), [h after each layer])``.
    """
    h = v4_embed_streams(ref, config, input_ids) if hidden is None else hidden
    per_layer = []
    for layer in ref["layers"][first_layer_idx:]:
        h = v4_block_forward(layer, config, h, input_ids)
        per_layer.append(h)
    with torch.no_grad():
        out = ref["norm"](ref["hc_head"](h))
    return out, per_layer


def v4_layer_weights(layer: dict, config) -> dict:
    """One layer's ``ref`` dict -> the per-layer entry of TtV4Transformer's ``state_dict["layers"]``."""
    return {
        "block": v4_block_state_dict(layer, config),
        "attn_reference": layer["attn"],
        "mhc_weights": v4_mhc_weights(layer),
    }


def v4_model_state_dict(ref: dict, config, first_layer_idx: int = 0, num_layers: Optional[int] = None) -> dict:
    """``ref`` -> TtV4Transformer's ``state_dict`` for layers ``[first_layer_idx, first_layer_idx + num_layers)``."""
    num_layers = len(ref["layers"]) - first_layer_idx if num_layers is None else num_layers
    head = ref["hc_head"]
    return {
        "embed_weight": ref["embed"].weight.detach(),
        "norm_weight": ref["norm"].weight.detach(),
        "hc_head": (head.hc_fn.detach(), head.hc_base.detach(), head.hc_scale.detach()),
        "layers": [
            v4_layer_weights(layer, config) for layer in ref["layers"][first_layer_idx : first_layer_idx + num_layers]
        ],
    }
