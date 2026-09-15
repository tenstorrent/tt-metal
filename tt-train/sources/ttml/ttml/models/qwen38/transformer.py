# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The Qwen3.8 text backbone: a hybrid Gated DeltaNet / Gated Attention stack.

All 64 layers share the same shape -- pre-norm token mixer, residual add,
pre-norm SwiGLU MLP, residual add -- and differ only in which mixer they use.
``full_attention_interval`` is 4, so layers 3, 7, 11, ... (0-indexed) are Gated
Attention and the other 48 are Gated DeltaNet.

Only the text path is built here. The checkpoint also carries a vision tower and
a one-layer MTP head; neither is part of the LoRA training graph, and the loader
skips both.
"""

from __future__ import annotations

from typing import Optional

import ttml
from ttml.modules import AbstractModuleBase, Embedding, LinearLayer, ModuleList, Parameter

from .attention import Qwen38GatedAttention
from .gated_deltanet import Qwen38GatedDeltaNet
from .checkpoint import recompute
from .parallel import make_column_linear, make_row_linear, tp_size

__all__ = [
    "Qwen38RMSNorm",
    "Qwen38MLP",
    "Qwen38Block",
    "Qwen38Transformer",
]


class Qwen38RMSNorm(AbstractModuleBase):
    """RMSNorm over the hidden dim, using the fused device op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.weight = Parameter(ttml.init.ones()((1, 1, 1, hidden_size)))

    def forward(self, hidden_states):
        return ttml.ops.rmsnorm.rmsnorm(hidden_states, self.weight.tensor, self.eps)


class Qwen38MLP(AbstractModuleBase):
    """SwiGLU MLP: ``down(silu(gate(x)) * up(x))``, via the fused device op.

    Under TP this is the standard Megatron pair: ``gate_proj`` and ``up_proj``
    are column-parallel (so the elementwise product stays local) and
    ``down_proj`` is row-parallel, costing a single all-reduce.
    """

    def __init__(self, config, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        init = ttml.init.normal(0.0, 0.02)
        # Megatron MLP: gate/up split the intermediate dim, down reduces over it
        # and all-reduces, so only one collective is needed per MLP.
        self.gate_proj = make_column_linear(config, hidden_size, intermediate_size, init)
        self.up_proj = make_column_linear(config, hidden_size, intermediate_size, init)
        self.down_proj = make_row_linear(config, intermediate_size, hidden_size, init)

        # The fused swiglu op is handed the three weights directly, which means
        # it bypasses the parallel layers' forward() and therefore their
        # collectives. Under TP the surrounding broadcast/all-reduce has to be
        # issued here instead, mirroring what ColumnParallelLinear and
        # RowParallelLinear(input_is_parallel=True) would have done.
        self.tp = tp_size(config)
        self.cluster_axis = ttml.mesh().axis_index(config.tp_axis_name) if self.tp > 1 else None

    def forward(self, x):
        if self.tp > 1:
            x = ttml.ops.distributed.broadcast(x, self.cluster_axis)
        out = ttml.ops.swiglu.swiglu(
            x,
            self.gate_proj.weight.tensor,
            self.down_proj.weight.tensor,
            self.up_proj.weight.tensor,
        )
        if self.tp > 1:
            # Each chip holds a partial sum over its slice of the intermediate
            # dim. noop_backward=True because the incoming gradient is already
            # correct per shard, matching RowParallelLinear's parallel-input case.
            out = ttml.ops.distributed.all_reduce(out, True, self.cluster_axis)
        return out


class Qwen38Block(AbstractModuleBase):
    """One decoder layer. The mixer is DeltaNet or Gated Attention per config.

    The DeltaNet mixer is stored as ``linear_attn`` and the attention mixer as
    ``self_attn``, mirroring the checkpoint's parameter names so the loader can
    address them without a layer-type lookup.
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_full_attention = config.is_full_attention(layer_idx)

        if self.is_full_attention:
            self.self_attn = Qwen38GatedAttention(config, layer_idx)
        else:
            self.linear_attn = Qwen38GatedDeltaNet(config, layer_idx)
        self.recompute_mixer = config.recompute_deltanet and not self.is_full_attention

        self.mlp = Qwen38MLP(config, config.hidden_size, config.intermediate_size)
        self.input_layernorm = Qwen38RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen38RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        mask: Optional[ttml.autograd.Tensor] = None,
        past_key_values=None,
        position_offset: int = 0,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.is_full_attention:
            hidden_states = self.self_attn(hidden_states, mask, past_key_values, position_offset)
        elif self.recompute_mixer:
            # The DeltaNet is inherently causal and carries no KV cache, so it
            # needs neither the mask nor the position offset.
            hidden_states = recompute(self.linear_attn, hidden_states)
        else:
            hidden_states = self.linear_attn(hidden_states)
        hidden_states = ttml.ops.binary.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return ttml.ops.binary.add(residual, hidden_states)


class Qwen38Transformer(AbstractModuleBase):
    """Embedding -> 64 hybrid blocks -> final norm -> LM head."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        init = ttml.init.normal(0.0, 0.02)

        # Column-parallel and deliberately ungathered: the full 248320-wide
        # logits are ~0.5 GB per 1024 tokens in bf16, so they are left sharded
        # and consumed by vocab_parallel_cross_entropy_loss.
        self.lm_head = make_column_linear(config, config.hidden_size, config.vocab_size, init)
        # The embedding table is padded to a tile boundary, as elsewhere in
        # tt-train. Qwen3.8's 248320 is already a multiple of 32.
        padded_vocab = (config.vocab_size + 31) // 32 * 32
        self.embed_tokens = Embedding(padded_vocab, config.hidden_size, weight_init=init)
        if config.tie_word_embeddings:
            self.embed_tokens.weight = self.lm_head.weight

        self.layers = ModuleList([Qwen38Block(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen38RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids,
        mask: Optional[ttml.autograd.Tensor] = None,
        past_key_values=None,
        position_offset: int = 0,
    ):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask, past_key_values, position_offset)
        hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)
