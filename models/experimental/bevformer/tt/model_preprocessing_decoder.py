# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device parameters for ``TtDetectionTransformerDecoder`` from the PyTorch reference.

Structure (heads, points, dims) is recorded next to the weights, so the TT modules take
it from here rather than from separate arguments.
"""

from types import SimpleNamespace

import torch

from models.experimental.bevformer.tt.model_preprocessing import (
    DEFAULT_DTYPE,
    preprocess_layer_norm_parameters,
    preprocess_linear_bias,
    preprocess_linear_weight,
    preprocess_ms_deformable_attention_parameters,
)


def _linear_params(weight, bias, device, dtype):
    return SimpleNamespace(
        weight=preprocess_linear_weight(weight, dtype=dtype, device=device),
        bias=preprocess_linear_bias(bias, dtype=dtype, device=device),
    )


def _self_attn_parameters(mha, device, dtype):
    """Q and K of ``nn.MultiheadAttention`` as one Linear, with Q pre-scaled by ``head_dim**-0.5``.

    Q and K both project ``query + query_pos``; V projects ``query`` alone.
    """
    head_dim = mha.embed_dim // mha.num_heads
    q_w, k_w, v_w = mha.in_proj_weight.chunk(3, dim=0)
    q_b, k_b, v_b = mha.in_proj_bias.chunk(3, dim=0)
    scale = head_dim**-0.5
    return SimpleNamespace(
        num_heads=mha.num_heads,
        qk_proj=_linear_params(torch.cat([q_w * scale, k_w]), torch.cat([q_b * scale, k_b]), device, dtype),
        v_proj=_linear_params(v_w, v_b, device, dtype),
        out_proj=_linear_params(mha.out_proj.weight, mha.out_proj.bias, device, dtype),
    )


def _cross_attn_parameters(msda, device, dtype):
    if msda.num_levels != 1:
        raise ValueError(f"the decoder cross-attention is single-level, got {msda.num_levels} levels")
    params = preprocess_ms_deformable_attention_parameters(msda, device=device, dtype=dtype)
    params.embed_dims = msda.embed_dims
    params.num_heads = msda.num_heads
    params.num_points = msda.num_points
    return params


def _layer_parameters(layer, device, dtype):
    ffn = layer.ffns[0].layers
    return SimpleNamespace(
        self_attn=_self_attn_parameters(layer.attentions[0].attn, device, dtype),
        cross_attn=_cross_attn_parameters(layer.attentions[1], device, dtype),
        ffn=SimpleNamespace(
            linear1=_linear_params(ffn[0][0].weight, ffn[0][0].bias, device, dtype),
            linear2=_linear_params(ffn[1].weight, ffn[1].bias, device, dtype),
        ),
        norms=[
            SimpleNamespace(**preprocess_layer_norm_parameters(norm, device=device, dtype=dtype))
            for norm in layer.norms
        ],
    )


def create_decoder_parameters(torch_model, device, dtype=DEFAULT_DTYPE):
    return SimpleNamespace(layers=[_layer_parameters(layer, device, dtype) for layer in torch_model.layers])


def create_reg_branch_parameters(reg_branches, device, dtype=DEFAULT_DTYPE):
    """Each branch is ``Linear-ReLU-Linear-ReLU-Linear``; returns its three Linears per branch."""
    return [
        [
            _linear_params(module.weight, module.bias, device, dtype)
            for module in branch
            if isinstance(module, torch.nn.Linear)
        ]
        for branch in reg_branches
    ]
