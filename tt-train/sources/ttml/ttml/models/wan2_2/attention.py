# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ttnn

import ttml
from ttml.autograd import Function
from ttml.modules import (
    AbstractModuleBase,
    ColumnParallelLinear,
    LinearLayer,
    Parameter,
    RowParallelLinear,
)

from .rope import apply as apply_rope


# TODO: Delete this Function once ttml gains split_heads, the single-tensor inverse of
# heads_fusion (bmijanovicTT)
# Issue: #53777
# heads_creation wants a fused (B, 1, S, 3E) qkv; Wan projects q/k/v separately.
class SplitHeads(Function):
    """(B, 1, S, heads*head_dim) -> (B, heads, S, head_dim), one tensor at a time.

    Not grouped_heads_creation: that requires Q and KV to share a sequence length, which
    cross-attention does not. num_kv_heads=0 means "split this tensor only".
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
    def __init__(self, dim: int, eps: float, tp_axis: int | None = None) -> None:
        super().__init__()
        self.eps = eps
        self.tp_axis = tp_axis
        self.weight = Parameter(ttml.init.ones()((1, 1, 1, dim)))

    def forward(self, x):
        if self.tp_axis is None:
            return ttml.ops.rmsnorm.rmsnorm(x, self.weight.tensor, self.eps)

        # TODO: Use ttml rmsnorm_distributed when implemented (bmijanovicTT)
        # Issue: #53775

        # REPLICATED, not the SHARDED default: scatter's backward already replicates, so the
        # default's reduce-scatter would inflate the gradient by exactly TP.
        full = ttml.ops.distributed.all_gather(x, 3, self.tp_axis, ttml.ops.distributed.GradOutputType.REPLICATED)
        normed = ttml.ops.rmsnorm.rmsnorm(full, self.weight.tensor, self.eps)
        return ttml.ops.distributed.scatter(normed, 3, self.tp_axis)


class WanAttention(AbstractModuleBase):
    def __init__(self, config, *, is_self: bool) -> None:
        super().__init__()
        dim = config.dim
        self.head_dim = config.head_dim
        self.is_self = is_self

        init = config.weight_init()
        tp_axis = None
        if config.use_tp:
            mesh = ttml.mesh()
            tp = mesh.axis_size("tp")
            if config.num_heads % tp:
                raise ValueError(f"num_heads {config.num_heads} is not divisible by TP size {tp}")
            tp_axis = mesh.axis_index("tp")
            self.num_heads = config.num_heads // tp
            col = dict(has_bias=True, weight_init=init, gather_output=False, axis_name="tp")
            self.to_q = ColumnParallelLinear(dim, dim, **col)
            self.to_k = ColumnParallelLinear(dim, dim, **col)
            self.to_v = ColumnParallelLinear(dim, dim, **col)
            self.to_out = RowParallelLinear(
                dim, dim, has_bias=True, weight_init=init, input_is_parallel=True, axis_name="tp"
            )
        else:
            self.num_heads = config.num_heads
            self.to_q = LinearLayer(dim, dim, True, weight_init=init)
            self.to_k = LinearLayer(dim, dim, True, weight_init=init)
            self.to_v = LinearLayer(dim, dim, True, weight_init=init)
            self.to_out = LinearLayer(dim, dim, True, weight_init=init)

        self.norm_q = _RMSNorm(dim, config.eps, tp_axis)
        self.norm_k = _RMSNorm(dim, config.eps, tp_axis)

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

        attn = ttml.ops.attention.scaled_dot_product_attention_composite(q, k, v, mask)
        return self.to_out(ttml.ops.multi_head_utils.heads_fusion(attn))
