# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-FileCopyrightText: Copyright 2020 The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.

# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import torch


@dataclass
class T5Config:
    vocab_size: int
    d_model: int
    d_ff: int
    d_kv: int
    num_heads: int
    num_layers: int
    relative_attention_num_buckets: int
    relative_attention_max_distance: int
    layer_norm_epsilon: float


# adapted from https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/t5/modeling_t5.py
class T5Encoder(torch.nn.Module):
    def __init__(self, config: T5Config) -> None:
        super().__init__()

        self.shared = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids)


# adapted from https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/t5/modeling_t5.py
class T5Stack(torch.nn.Module):
    def __init__(self, config: T5Config) -> None:
        super().__init__()

        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.block = torch.nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(d_model=config.d_model, eps=config.layer_norm_epsilon)

        self._relative_attention_num_buckets = config.relative_attention_num_buckets
        self._relative_attention_max_distance = config.relative_attention_max_distance

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _batch_size, seq_length = input_ids.size()

        inputs_embeds = self.embed_tokens(input_ids)

        relative_attention_bias = self.block[0].layer[0].SelfAttention.relative_attention_bias
        position_bias = _compute_bias(
            seq_length=seq_length,
            device=input_ids.device,
            relative_attention_num_buckets=self._relative_attention_num_buckets,
            relative_attention_max_distance=self._relative_attention_max_distance,
            relative_attention_bias=relative_attention_bias,
        )
        position_bias = position_bias[:, :, -seq_length:, :]

        hidden_states = inputs_embeds

        for layer_module in self.block:
            hidden_states = layer_module(hidden_states, position_bias=position_bias)

        return self.final_layer_norm(hidden_states)


# adapted from https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/t5/modeling_t5.py
class T5Block(torch.nn.Module):
    def __init__(self, config: T5Config, *, has_relative_attention_bias: bool) -> None:
        super().__init__()

        self.layer = torch.nn.ModuleList(
            [
                T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias),
                T5LayerFF(
                    d_model=config.d_model,
                    d_ff=config.d_ff,
                    eps=config.layer_norm_epsilon,
                ),
            ]
        )

    def forward(self, hidden_states: torch.Tensor, position_bias: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer[0](hidden_states, position_bias=position_bias)
        return self.layer[1](hidden_states)


# adapted from https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/t5/modeling_t5.py
class T5LayerSelfAttention(torch.nn.Module):
    def __init__(self, config: T5Config, *, has_relative_attention_bias: bool) -> None:
        super().__init__()

        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(d_model=config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, x: torch.Tensor, *, position_bias: torch.Tensor) -> torch.Tensor:
        normed = self.layer_norm(x)
        attn = self.SelfAttention(normed, position_bias=position_bias)
        return x + attn


# adapted from https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/t5/modeling_t5.py
class T5Attention(torch.nn.Module):
    def __init__(self, config: T5Config, *, has_relative_attention_bias: bool) -> None:
        super().__init__()
        self._config = config
        self._kv_dim = config.d_kv
        self._num_heads = config.num_heads
        self.inner_dim = self._num_heads * self._kv_dim

        self.q = torch.nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.k = torch.nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.v = torch.nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.o = torch.nn.Linear(self.inner_dim, config.d_model, bias=False)

        if has_relative_attention_bias:
            self.relative_attention_bias = torch.nn.Embedding(config.relative_attention_num_buckets, self._num_heads)

    def forward(self, hidden_states: torch.Tensor, position_bias: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape

        query_states = self.q(hidden_states)
        key_states = self.k(hidden_states)
        value_states = self.v(hidden_states)

        query_states = query_states.view(batch_size, -1, self._num_heads, self._kv_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self._num_heads, self._kv_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self._num_heads, self._kv_dim).transpose(1, 2)

        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        scores += position_bias
        attn_weights = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.inner_dim)
        return self.o(attn_output)


# adapted from https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/t5/modeling_t5.py
class T5LayerFF(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, eps: float) -> None:
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(d_model=d_model, d_ff=d_ff)
        self.layer_norm = T5LayerNorm(d_model=d_model, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fw = self.layer_norm(x)
        fw = self.DenseReluDense(fw)
        return x + fw


# adapted from https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/t5/modeling_t5.py
class T5DenseGatedActDense(torch.nn.Module):
    def __init__(self, *, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.wi_0 = torch.nn.Linear(d_model, d_ff, bias=False)
        self.wi_1 = torch.nn.Linear(d_model, d_ff, bias=False)
        self.wo = torch.nn.Linear(d_ff, d_model, bias=False)
        self.act = NewGELUActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gelu = self.act(self.wi_0(x))
        linear = self.wi_1(x)
        x = gelu * linear
        return self.wo(x)


# adapted from https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/t5/modeling_t5.py
class T5LayerNorm(torch.nn.Module):
    def __init__(self, *, d_model: int, eps: float) -> None:
        super().__init__()

        self.weight = torch.nn.Parameter(torch.ones([d_model]))
        self._variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self._variance_epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.to(self.weight.dtype)

        return self.weight * x


# adapted from https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/t5/modeling_t5.py
def _relative_position_bucket(relative_position: torch.Tensor, num_buckets: int, max_distance: int) -> torch.Tensor:
    num_buckets //= 2

    relative_buckets = (relative_position > 0).to(torch.long) * num_buckets
    relative_position = torch.abs(relative_position)

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large,
        torch.full_like(relative_position_if_large, num_buckets - 1),
    )

    relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


# adapted from https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/models/t5/modeling_t5.py
def _compute_bias(
    *,
    seq_length: int,
    device: torch.device,
    relative_attention_num_buckets: int,
    relative_attention_max_distance: int,
    relative_attention_bias: torch.nn.Module,
) -> torch.Tensor:
    context_position = torch.arange(seq_length, device=device)[:, None]
    memory_position = torch.arange(seq_length, device=device)[None, :]
    relative_position = memory_position - context_position

    relative_position_bucket = _relative_position_bucket(
        relative_position,
        num_buckets=relative_attention_num_buckets,
        max_distance=relative_attention_max_distance,
    )

    output = relative_attention_bias(relative_position_bucket)
    return output.permute([2, 0, 1]).unsqueeze(0)


# adapted from https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/activations.py
class NewGELUActivation(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = math.sqrt(2.0 / math.pi)
        y = 0.044715 * torch.pow(x, 3.0) + x
        return 0.5 * x * (1.0 + torch.tanh(c * y))
