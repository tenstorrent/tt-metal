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
from models.common.lightweightmodule import LightweightModule


def _to_device(torch_tensor, device):
    """torch -> ttnn bf16 tile tensor on device."""
    return ttnn.from_torch(
        torch_tensor.to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )


class TtXttsGptBlock(LightweightModule):
    def __init__(
        self,
        state_dict,
        device,
        layer_idx=0,
    ):
        super().__init__()
        self.device = device
        self.layer_idx = layer_idx

        prefix = f"gpt.gpt.h.{layer_idx}."

        # Load layer norm parameters
        self.ln_1_weight = _to_device(state_dict[prefix + "ln_1.weight"], device)
        self.ln_1_bias = _to_device(state_dict[prefix + "ln_1.bias"], device)
        self.ln_2_weight = _to_device(state_dict[prefix + "ln_2.weight"], device)
        self.ln_2_bias = _to_device(state_dict[prefix + "ln_2.bias"], device)

        # Load attention parameters
        self.attn_c_attn_weight = _to_device(state_dict[prefix + "attn.c_attn.weight"], device)
        self.attn_c_attn_bias = _to_device(state_dict[prefix + "attn.c_attn.bias"], device)
        self.attn_c_proj_weight = _to_device(state_dict[prefix + "attn.c_proj.weight"], device)
        self.attn_c_proj_bias = _to_device(state_dict[prefix + "attn.c_proj.bias"], device)

        # Load MLP parameters
        self.mlp_c_fc_weight = _to_device(state_dict[prefix + "mlp.c_fc.weight"], device)
        self.mlp_c_fc_bias = _to_device(state_dict[prefix + "mlp.c_fc.bias"], device)
        self.mlp_c_proj_weight = _to_device(state_dict[prefix + "mlp.c_proj.weight"], device)
        self.mlp_c_proj_bias = _to_device(state_dict[prefix + "mlp.c_proj.bias"], device)

    def _split_heads(self, x):  # [b, s, hidden] -> [b, heads, s, head_dim]
        b, s, _ = x.shape
        x = ttnn.reshape(x, (b, s, NUM_HEADS, HEAD_DIM))
        return ttnn.permute(x, (0, 2, 1, 3))

    def _merge_heads(self, x):  # [b, heads, s, head_dim] -> [b, s, hidden]
        b, _, s, _ = x.shape
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.reshape(x, (b, s, HIDDEN_SIZE))

    def _attention(self, x):
        qkv = ttnn.linear(x, self.attn_c_attn_weight, bias=self.attn_c_attn_bias)
        b, s = qkv.shape[0], qkv.shape[1]
        q = ttnn.slice(qkv, [0, 0, 0], [b, s, HIDDEN_SIZE])
        k = ttnn.slice(qkv, [0, 0, HIDDEN_SIZE], [b, s, 2 * HIDDEN_SIZE])
        v = ttnn.slice(qkv, [0, 0, 2 * HIDDEN_SIZE], [b, s, 3 * HIDDEN_SIZE])
        ttnn.deallocate(qkv)

        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = self._merge_heads(attn)
        return ttnn.linear(out, self.attn_c_proj_weight, bias=self.attn_c_proj_bias)

    def _mlp(self, x):
        h = ttnn.linear(x, self.mlp_c_fc_weight, bias=self.mlp_c_fc_bias)
        h = ttnn.gelu(h)  # tanh-approx GELU ~= GPT-2 "gelu_new" (validated by PCC)
        return ttnn.linear(h, self.mlp_c_proj_weight, bias=self.mlp_c_proj_bias)

    def forward(self, x):
        """Run one XTTS GPT decoder block. ``x`` is ``[batch, seq, hidden]`` on device."""
        h = ttnn.layer_norm(x, weight=self.ln_1_weight, bias=self.ln_1_bias, epsilon=LAYER_NORM_EPS)
        x = ttnn.add(x, self._attention(h))
        h = ttnn.layer_norm(x, weight=self.ln_2_weight, bias=self.ln_2_bias, epsilon=LAYER_NORM_EPS)
        return ttnn.add(x, self._mlp(h))
