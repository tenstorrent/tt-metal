# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device parameters for ``TtDetectionTransformerDecoder`` from the PyTorch reference.

Structure (heads, points, dims, the cross-attention config) is recorded next to the
weights, so the TT modules take it from here rather than from separate arguments.
"""

from types import SimpleNamespace

import torch

from models.experimental.bevformer.config.decoder_config import REG_XY, REG_Z
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
    """The reference module's parameters and config."""
    if msda.num_levels != 1:
        raise ValueError(f"the decoder cross-attention is single-level, got {msda.num_levels} levels")
    params = preprocess_ms_deformable_attention_parameters(msda, device=device, dtype=dtype)
    params.config = msda.config
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
    """Parameters for a single ``TtDetectionTransformerDecoder``.

    Single use: the decoder's constructor folds the BEV size into each layer's
    ``cross_attn.sampling_offsets`` and frees the original, so every decoder instance (and
    every BEV size) needs its own call.
    """
    return SimpleNamespace(layers=[_layer_parameters(layer, device, dtype) for layer in torch_model.layers])


def create_reg_branch_parameters(reg_branches, device, dtype=DEFAULT_DTYPE):
    """The three Linears of each ``Linear-ReLU-Linear-ReLU-Linear`` branch, for the decoder's
    reference-point refinement.

    The last Linear keeps only the box-code rows the refinement reads, (x, y, z) in that
    order (``REG_XY``, ``REG_Z``), so the decoder adds its output to the points' logits as is.
    """
    code_size = reg_branches[0][-1].out_features
    rows = list(range(code_size))[REG_XY] + list(range(code_size))[REG_Z]
    branches = []
    for branch in reg_branches:
        first, second, last = (module for module in branch if isinstance(module, torch.nn.Linear))
        branches.append(
            [
                _linear_params(first.weight, first.bias, device, dtype),
                _linear_params(second.weight, second.bias, device, dtype),
                _linear_params(last.weight[rows], last.bias[rows], device, dtype),
            ]
        )
    return branches
