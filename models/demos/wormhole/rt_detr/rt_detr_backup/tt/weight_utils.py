# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch

import ttnn


def fold_bn_to_conv(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
    std = torch.sqrt(bn_var + eps)
    scale = bn_w / std
    fused_w = conv_w * scale.view(-1, 1, 1, 1)
    fused_b = (conv_b - bn_mean) * scale + bn_b
    return fused_w, fused_b


def _to_tt(t, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


class Params:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _linear(w, b, device):
    return Params(
        weight=_to_tt(w.T.contiguous(), device, dtype=ttnn.bfloat8_b),
        bias=_to_tt(b.reshape(1, 1, 1, -1), device, dtype=ttnn.bfloat16),
    )


def _norm(w, b, device):
    return Params(
        weight=_to_tt(w.reshape(1, 1, 1, -1), device, dtype=ttnn.bfloat16),
        bias=_to_tt(b.reshape(1, 1, 1, -1), device, dtype=ttnn.bfloat16),
    )


def get_tt_parameters(device, model):
    if hasattr(model, "encoder"):
        src_layers = model.encoder.encoder[0].layers
    else:
        src_layers = model.layers

    layers = []
    for layer in src_layers:
        attn = layer.self_attn
        d = attn.embed_dim
        qkv_w = attn.in_proj_weight
        qkv_b = attn.in_proj_bias

        def qkv_split(start, end):
            return Params(
                weight=_to_tt(qkv_w[start:end, :].T.contiguous(), device, dtype=ttnn.bfloat8_b),
                bias=_to_tt(qkv_b[start:end].reshape(1, 1, 1, -1), device, dtype=ttnn.bfloat16),
            )

        layers.append(
            Params(
                self_attn=Params(
                    q=qkv_split(0, d),
                    k=qkv_split(d, 2 * d),
                    v=qkv_split(2 * d, 3 * d),
                    out_proj=_linear(attn.out_proj.weight, attn.out_proj.bias, device),
                ),
                linear1=_linear(layer.linear1.weight, layer.linear1.bias, device),
                linear2=_linear(layer.linear2.weight, layer.linear2.bias, device),
                norm1=_norm(layer.norm1.weight, layer.norm1.bias, device),
                norm2=_norm(layer.norm2.weight, layer.norm2.bias, device),
            )
        )

    return Params(layers=layers)
