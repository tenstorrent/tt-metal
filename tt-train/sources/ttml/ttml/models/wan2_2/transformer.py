# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Wan transformer block: modulated self-attention, cross-attention, modulated feedforward."""

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

from .attention import WanAttention

_MOD_CHUNKS = 6


# TODO: Delete this Function once ttml.ops.unary.gelu takes a variant (bmijanovicTT)
# Issue: #
# ttml hardcodes the erf form both ways: ttnn::gelu with no variant forward, gelu_bw with
# approx_mode "none" backward.
class GeluTanh(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ttnn.gelu(x.get_value(), variant=ttnn.GeluVariant.Tanh)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return ttnn.experimental.gelu_bw(grad_output, x.get_value(), approximate="tanh")


class WanFeedForward(AbstractModuleBase):
    def __init__(self, config) -> None:
        super().__init__()
        init = config.weight_init()
        if config.use_tp:
            tp = ttml.mesh().axis_size("tp")
            if config.ffn_dim % tp:
                raise ValueError(f"ffn_dim {config.ffn_dim} is not divisible by TP size {tp}")
            # No norm between them, so the intermediate stays sharded end to end.
            self.ff1 = ColumnParallelLinear(
                config.dim,
                config.ffn_dim,
                has_bias=True,
                weight_init=init,
                gather_output=False,
                axis_name="tp",
            )
            self.ff2 = RowParallelLinear(
                config.ffn_dim,
                config.dim,
                has_bias=True,
                weight_init=init,
                input_is_parallel=True,
                axis_name="tp",
            )
        else:
            self.ff1 = LinearLayer(config.dim, config.ffn_dim, True, weight_init=init)
            self.ff2 = LinearLayer(config.ffn_dim, config.dim, True, weight_init=init)

    def forward(self, x):
        return self.ff2(GeluTanh.apply(self.ff1(x)))


class WanLayerNorm(AbstractModuleBase):
    """Affine LayerNorm; ttml's layernorm fixes eps at 1e-6, which is Wan's value."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = Parameter(ttml.init.ones()((1, 1, 1, dim)))
        self.bias = Parameter(ttml.init.zeros()((1, 1, 1, dim)))

    def forward(self, x):
        return ttml.ops.layernorm.layernorm(x, self.weight.tensor, self.bias.tensor)


class WanTransformerBlock(AbstractModuleBase):
    def __init__(self, config) -> None:
        super().__init__()
        self.attn1 = WanAttention(config, is_self=True)
        self.attn2 = WanAttention(config, is_self=False)
        self.norm2 = WanLayerNorm(config.dim) if config.cross_attn_norm else None
        self.ffn = WanFeedForward(config)
        self.scale_shift_table = Parameter(ttml.init.zeros()((1, 1, _MOD_CHUNKS, config.dim)))

    def _modulation(self, temb):
        """Six modulation tensors from the timestep embedding. Frozen, so raw ttnn."""
        shifted = ttnn.add(self.scale_shift_table.tensor.get_value(), temb.get_value())
        shift, scale, gate, c_shift, c_scale, c_gate = ttnn.chunk(shifted, _MOD_CHUNKS, dim=2)

        def const(value):
            return ttml.autograd.create_tensor(value, False)

        return (
            const(ttnn.add(scale, 1.0)),
            const(shift),
            const(gate),
            const(ttnn.add(c_scale, 1.0)),
            const(c_shift),
            const(c_gate),
        )

    def forward(self, x, mask, prompt, temb, rope_params):
        # mask is second so memory_efficient_runner can call this as (input, mask, *extras).
        gamma1, beta1, gate1, gamma3, beta3, gate3 = self._modulation(temb)

        h = ttml.ops.layernorm.layernorm(x, gamma1, beta1)
        x = x + self.attn1(h, rope_params=rope_params) * gate1

        h = self.norm2(x) if self.norm2 is not None else x
        x = x + self.attn2(h, context=prompt, mask=mask)

        h = ttml.ops.layernorm.layernorm(x, gamma3, beta3)
        x = x + self.ffn(h) * gate3
        return x
