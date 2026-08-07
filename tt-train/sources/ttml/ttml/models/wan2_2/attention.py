# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Wan self- and cross-attention. Tensors are (B, 1, S, dim); heads are (B, heads, S, head_dim)."""

from __future__ import annotations

import ttnn

import ttml
from ttml.autograd import Function
from ttml.modules import AbstractModuleBase, LinearLayer, Parameter

from .rope import apply as apply_rope


class SplitHeads(Function):
    """(B, 1, S, heads*head_dim) -> (B, heads, S, head_dim) for one tensor.

    ttml's grouped_heads_creation requires Q and KV to share a sequence length, which is
    false for cross-attention (image tokens vs caption tokens), so each tensor is split on
    its own. Uses the same ttnn kernels ttml's own head ops use -- reshape plus permute on a
    tiled tensor is not a supported path. num_kv_heads=0 means "split this tensor only".
    """

    @staticmethod
    def forward(ctx, x, num_heads):
        heads, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            x.get_value(), num_heads=num_heads, num_kv_heads=0, transpose_k_heads=False
        )
        return heads

    @staticmethod
    def backward(ctx, grad_output):
        return ttnn.experimental.nlp_concat_heads(grad_output)


class _RMSNorm(AbstractModuleBase):
    """RMSNorm over the whole projected dim -- Wan normalises across heads, before the split."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = Parameter(ttml.init.ones()((1, 1, 1, dim)))

    def forward(self, x):
        return ttml.ops.rmsnorm.rmsnorm(x, self.weight.tensor, self.eps)


class WanAttention(AbstractModuleBase):
    def __init__(self, config, *, is_self: bool) -> None:
        super().__init__()
        dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.is_self = is_self

        init = config.weight_init()
        self.to_q = LinearLayer(dim, dim, True, weight_init=init)
        self.to_k = LinearLayer(dim, dim, True, weight_init=init)
        self.to_v = LinearLayer(dim, dim, True, weight_init=init)
        self.to_out = LinearLayer(dim, dim, True, weight_init=init)

        self.norm_q = _RMSNorm(dim, config.eps)
        self.norm_k = _RMSNorm(dim, config.eps)

    def forward(self, x, context=None, rope_params=None, mask=None):
        source = x if self.is_self else context
        if source is None:
            raise ValueError("cross-attention requires a context tensor")

        q = SplitHeads.apply(self.norm_q(self.to_q(x)), self.num_heads)
        k = SplitHeads.apply(self.norm_k(self.to_k(source)), self.num_heads)
        v = SplitHeads.apply(self.to_v(source), self.num_heads)

        if self.is_self:
            if rope_params is None:
                raise ValueError("self-attention requires rope_params")
            q = apply_rope(q, rope_params)
            k = apply_rope(k, rope_params)

        # Wan attends bidirectionally over patches. The fused kernel defaults its mask type to
        # Causal when no mask is passed, which silently truncates attention, so use the
        # composite path -- it applies no mask unless given one. It also handles the unequal
        # q/k lengths of cross-attention.
        attn = ttml.ops.attention.scaled_dot_product_attention_composite(q, k, v, mask)
        return self.to_out(ttml.ops.multi_head_utils.heads_fusion(attn))
