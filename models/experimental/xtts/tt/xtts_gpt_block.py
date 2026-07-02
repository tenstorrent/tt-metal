# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of a single XTTS-v2 GPT decoder block.

Mirrors ``reference/xtts_gpt_block.py`` (a HuggingFace ``GPT2Block``):

    h = x + attn(ln_1(x))          # causal multi-head self-attention
    y = h + mlp(ln_2(h))           # c_fc -> gelu -> c_proj

Weight-layout notes:
  * GPT-2 uses ``Conv1D``, whose weight is stored ``[in, out]`` — already the
    layout ``ttnn.linear`` expects (y = x @ W + b), so NO transpose is needed.
  * Attention is causal with scale ``1/sqrt(head_dim)`` — matches the defaults
    of ``ttnn.transformer.scaled_dot_product_attention``.
"""

import torch
import ttnn

from models.experimental.xtts.reference.xtts_gpt_block import (
    HEAD_DIM,
    HIDDEN_SIZE,
    LAYER_NORM_EPS,
    NUM_HEADS,
)


def _to_device(torch_tensor, device):
    """torch -> ttnn bf16 tile tensor on device."""
    return ttnn.from_torch(
        torch_tensor.to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )


def load_gpt_block_parameters(state_dict, device, layer_idx=0):
    """Build the on-device parameter dict for one GPT block.

    ``state_dict`` is the full XTTS checkpoint state dict; the block's weights
    live under ``gpt.gpt.h.{layer_idx}.``. Conv1D weights are loaded as-is.
    """
    prefix = f"gpt.gpt.h.{layer_idx}."

    def w(name):
        return _to_device(state_dict[prefix + name], device)

    return {
        "ln_1.weight": w("ln_1.weight"),
        "ln_1.bias": w("ln_1.bias"),
        "attn.c_attn.weight": w("attn.c_attn.weight"),  # [1024, 3072] Conv1D (no transpose)
        "attn.c_attn.bias": w("attn.c_attn.bias"),
        "attn.c_proj.weight": w("attn.c_proj.weight"),  # [1024, 1024]
        "attn.c_proj.bias": w("attn.c_proj.bias"),
        "ln_2.weight": w("ln_2.weight"),
        "ln_2.bias": w("ln_2.bias"),
        "mlp.c_fc.weight": w("mlp.c_fc.weight"),  # [1024, 4096]
        "mlp.c_fc.bias": w("mlp.c_fc.bias"),
        "mlp.c_proj.weight": w("mlp.c_proj.weight"),  # [4096, 1024]
        "mlp.c_proj.bias": w("mlp.c_proj.bias"),
    }


def _split_heads(x):  # [b, s, hidden] -> [b, heads, s, head_dim]
    b, s, _ = x.shape
    x = ttnn.reshape(x, (b, s, NUM_HEADS, HEAD_DIM))
    return ttnn.permute(x, (0, 2, 1, 3))


def _merge_heads(x):  # [b, heads, s, head_dim] -> [b, s, hidden]
    b, _, s, _ = x.shape
    x = ttnn.permute(x, (0, 2, 1, 3))
    return ttnn.reshape(x, (b, s, HIDDEN_SIZE))


def _attention(x, params):
    qkv = ttnn.linear(x, params["attn.c_attn.weight"], bias=params["attn.c_attn.bias"])
    b, s = qkv.shape[0], qkv.shape[1]
    q = ttnn.slice(qkv, [0, 0, 0], [b, s, HIDDEN_SIZE])
    k = ttnn.slice(qkv, [0, 0, HIDDEN_SIZE], [b, s, 2 * HIDDEN_SIZE])
    v = ttnn.slice(qkv, [0, 0, 2 * HIDDEN_SIZE], [b, s, 3 * HIDDEN_SIZE])
    ttnn.deallocate(qkv)

    q, k, v = _split_heads(q), _split_heads(k), _split_heads(v)
    attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = _merge_heads(attn)
    return ttnn.linear(out, params["attn.c_proj.weight"], bias=params["attn.c_proj.bias"])


def _mlp(x, params):
    h = ttnn.linear(x, params["mlp.c_fc.weight"], bias=params["mlp.c_fc.bias"])
    h = ttnn.gelu(h)  # tanh-approx GELU ~= GPT-2 "gelu_new" (validated by PCC)
    return ttnn.linear(h, params["mlp.c_proj.weight"], bias=params["mlp.c_proj.bias"])


def xtts_gpt_block(x, params):
    """Run one XTTS GPT decoder block. ``x`` is ``[batch, seq, hidden]`` on device."""
    h = ttnn.layer_norm(x, weight=params["ln_1.weight"], bias=params["ln_1.bias"], epsilon=LAYER_NORM_EPS)
    x = ttnn.add(x, _attention(h, params))
    h = ttnn.layer_norm(x, weight=params["ln_2.weight"], bias=params["ln_2.bias"], epsilon=LAYER_NORM_EPS)
    return ttnn.add(x, _mlp(h, params))
