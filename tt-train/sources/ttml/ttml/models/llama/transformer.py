# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama transformer block."""

from __future__ import annotations

from typing import Callable, Optional

import ttml
from ttml.modules import AbstractModuleBase, Parameter, RunMode, LinearLayer, ColumnParallelLinear, RowParallelLinear

from .gqattn import GroupedQueryAttention


def compute_swiglu_intermediate_size(hidden_size: int, multiple_of: int = 256) -> int:
    """Compute the default MLP intermediate size for Llama.

    Meta's Llama uses SwiGLU which has 3 matrices (w1, w2, w3) instead of 2 in a
    standard MLP. To match the parameter count of a conventional 4x MLP, the
    intermediate size is scaled to 2/3 of 4*hidden = 8/3*hidden, then rounded up
    to ``multiple_of`` for hardware alignment.
    """
    unrounded = (4 * hidden_size * 2) // 3
    return ((unrounded + multiple_of - 1) // multiple_of) * multiple_of


class RMSNormLayer(AbstractModuleBase):
    def __init__(
        self,
        features: int,
        epsilon: float = 1e-5,
        use_composite: bool = False,
    ) -> None:
        super().__init__()

        self.epsilon = epsilon
        self.use_composite = use_composite

        gamma_shape = (1, 1, 1, features)
        self.gamma = Parameter(ttml.init.ones()(gamma_shape))

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of RMSNorm.

        Args:
            x: Input tensor

        Returns:
            Normalized output tensor
        """

        if self.use_composite:
            rmsnorm_op = ttml.ops.rmsnorm.rmsnorm_composite
        else:
            rmsnorm_op = ttml.ops.rmsnorm.rmsnorm

        return rmsnorm_op(x, self.gamma.tensor, self.epsilon)


class LlamaMLP(AbstractModuleBase):
    """Llama-style MLP (feed-forward) layer."""

    def __init__(
        self,
        embedding_size: int,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.0,
        use_tp: bool = False,
        down_proj_init: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.embedding_size = embedding_size
        self.dropout_prob = dropout
        # Distinct mask per device only when each holds distinct data. Too coarse under DP+TP:
        # ttnn offsets the seed by flat device id, so sharing within TP also shares across DP.
        self.dropout_per_device_seed = not use_tp

        if intermediate_size is None:
            intermediate_size = compute_swiglu_intermediate_size(embedding_size)

        # Fused gate+up: one [2*I, E] weight, rows [0:I) = gate, [I:2*I) = up.
        gate_up_size = 2 * intermediate_size

        # Each device's gate|up half is I/tp wide and must stay tile-aligned, or the packed
        # SwiGLU op's two halves straddle a tile boundary.
        tp_size = ttml.mesh().axis_size("tp") if use_tp else 1
        if intermediate_size % tp_size != 0:
            raise ValueError(
                f"intermediate_size ({intermediate_size}) must be divisible by the tensor-parallel size ({tp_size})"
            )
        if (intermediate_size // tp_size) % 32 != 0:
            raise ValueError(
                f"intermediate_size per device ({intermediate_size // tp_size}) must be a "
                f"multiple of 32 so the packed gate|up halves stay tile-aligned"
            )

        if use_tp:
            self.w_gate_up = ColumnParallelLinear(
                embedding_size,
                gate_up_size,
                has_bias=False,
                gather_output=False,
                axis_name="tp",
            )
            self.w2 = RowParallelLinear(
                intermediate_size,
                embedding_size,
                has_bias=False,
                weight_init=down_proj_init,
                input_is_parallel=True,
                axis_name="tp",
            )
        else:
            self.w_gate_up = LinearLayer(
                embedding_size,
                gate_up_size,
                False,
            )
            self.w2 = LinearLayer(
                intermediate_size,
                embedding_size,
                False,
                weight_init=down_proj_init,
            )

    def forward(self, input: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of MLP.

        Args:
            x: Input tensor

        Returns:
            Output tensor after MLP
        """
        gu = self.w_gate_up(input)
        h = ttml.ops.swiglu_packed.swiglu_packed(gu)
        x = self.w2(h)

        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            x = ttml.ops.dropout.dropout(x, self.dropout_prob, use_per_device_seed=self.dropout_per_device_seed)

        return x


class LlamaBlock(AbstractModuleBase):
    """Pre-norm residual transformer block (attention + MLP)."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rope_params: ttml.ops.rope.RotaryEmbeddingParams,
        attention_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        intermediate_size: Optional[int] = None,
        attention_bias: bool = False,
        use_tp: bool = False,
        out_proj_init: Optional[Callable] = None,
        down_proj_init: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.mlp = LlamaMLP(
            hidden_size,
            intermediate_size,
            mlp_dropout,
            use_tp=use_tp,
            down_proj_init=down_proj_init,
        )
        self.attention_norm = RMSNormLayer(hidden_size)
        self.mlp_norm = RMSNormLayer(hidden_size)
        self.attention = GroupedQueryAttention(
            embedding_size=hidden_size,
            num_heads=num_attention_heads,
            num_groups=num_key_value_heads,
            dropout=attention_dropout,
            rope_params=rope_params,
            bias_linears=attention_bias,
            use_tp=use_tp,
            out_proj_init=out_proj_init,
        )

    def forward(
        self,
        input: ttml.autograd.Tensor,
        mask: ttml.autograd.Tensor,
        kv_cache: Optional[ttml.models.KvCache] = None,
        layer_idx: Optional[int] = None,
        new_tokens: Optional[int] = None,
    ) -> ttml.autograd.Tensor:
        residual = input
        h = self.attention_norm(input)
        h = self.attention(h, mask, kv_cache, layer_idx, new_tokens)
        h = ttml.ops.binary.add(h, residual)

        residual = h
        x = self.mlp_norm(h)
        x = self.mlp(x)
        x = ttml.ops.binary.add(x, residual)

        return x
