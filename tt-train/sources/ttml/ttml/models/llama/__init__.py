# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict

import numpy as np
import ml_dtypes

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, ModuleList, LinearLayer

from .. import RunnerType, WeightTyingType, memory_efficient_runner
from .embedding import Embedding
from .transformer import LlamaBlock, RMSNormLayer


@dataclass(frozen=True)
class LlamaRopeScalingConfig:
    scaling_factor: float = 0.0  # 0.0 means no scaling
    high_freq_factor: float = 4.0
    low_freq_factor: float = 1.0
    original_context_length: int = 0  # 0 means no scaling


@dataclass(frozen=True)
class LlamaConfig:
    hidden_size: int = 384
    intermediate_size: Optional[int] = None
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    num_key_value_heads: int = 2
    vocab_size: int = 256
    max_position_embeddings: int = 256
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_dropout: float = 0.0
    runner_type: RunnerType = RunnerType.Default
    weight_tying: WeightTyingType = WeightTyingType.Disabled
    rope_scaling: LlamaRopeScalingConfig = field(default_factory=LlamaRopeScalingConfig)

    def __post_init__(self):
        if self.max_position_embeddings % 32 != 0:
            raise ValueError(
                "Max position embeddings must be divisible by 32 due to current limitations in tensor. "
                f"Provided max_position_embeddings={self.max_position_embeddings}"
            )
        if self.hidden_size % 32 != 0:
            raise ValueError(
                "Hidden size must be divisible by 32 due to current limitations in tensor. "
                f"Provided hidden_size={self.hidden_size}"
            )
        if self.num_attention_heads <= 0:
            raise ValueError(
                "Number of attention heads must be a positive integer. "
                f"Provided num_attention_heads={self.num_attention_heads}"
            )
        if self.num_key_value_heads <= 0:
            raise ValueError(
                "Number of key/value heads must be a positive integer. "
                f"Provided num_key_value_heads={self.num_key_value_heads}"
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "Hidden size must be divisible by the number of attention heads. "
                f"Provided hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads}"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "Number of attention heads must be divisible by the number of key/value heads. "
                f"Provided num_attention_heads={self.num_attention_heads}, num_key_value_heads={self.num_key_value_heads}"
            )


def initialize_parameters(parameters: Dict[str, ttml.autograd.Tensor]) -> None:
    for name, tensor in parameters.items():
        shape = tensor.shape()

        if "weight" in name:
            # Re-initialize weights with normal(0, 0.02)
            weight_np = np.random.normal(0.0, 0.02, size=shape).astype(
                ml_dtypes.bfloat16
            )
            new_tensor = ttml.autograd.Tensor.from_numpy(
                weight_np, layout=ttnn.Layout.TILE
            )
            tensor.assign(new_tensor)
        elif "bias" in name:
            # Re-initialize biases with 0
            bias_np = np.zeros(shape, dtype=ml_dtypes.bfloat16)
            new_tensor = ttml.autograd.Tensor.from_numpy(
                bias_np, layout=ttnn.Layout.TILE
            )
            tensor.assign(new_tensor)


class Llama(AbstractModuleBase):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()

        self.config = config

        self.fc = LinearLayer(config.hidden_size, config.vocab_size, False)

        vocab_size_divisible_by_32 = (config.vocab_size + 31) // 32 * 32
        self.tok_emb = Embedding(vocab_size_divisible_by_32, config.hidden_size)

        if config.weight_tying == ttml.models.WeightTyingType.Enabled:
            self.tok_emb.weight = self.fc.get_weight()

        head_dim = config.hidden_size // config.num_attention_heads

        rope_scaling_params = ttml.ops.rope.RopeScalingParams()
        if (
            config.rope_scaling.scaling_factor != 0.0
            and config.rope_scaling.original_context_length != 0
        ):
            rope_scaling_params.scaling_factor = config.rope_scaling.scaling_factor
            rope_scaling_params.high_freq_factor = config.rope_scaling.high_freq_factor
            rope_scaling_params.low_freq_factor = config.rope_scaling.low_freq_factor
            rope_scaling_params.original_context_length = (
                config.rope_scaling.original_context_length
            )

        rope_params = ttml.ops.rope.build_rope_params(
            config.max_position_embeddings,
            head_dim,
            config.rope_theta,
            rope_scaling_params,
        )

        # Transformer blocks (ModuleList auto-registers all blocks)
        self.blocks = ModuleList(
            [
                LlamaBlock(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    rope_params=rope_params,
                    attention_dropout=config.attention_dropout,
                    mlp_dropout=config.mlp_dropout,
                    intermediate_size=config.intermediate_size,
                    attention_bias=config.attention_bias,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.ln_fc = RMSNormLayer(config.hidden_size)
        initialize_parameters(self.parameters())

    def forward(
        self,
        input: ttml.autograd.Tensor,
        mask: ttml.autograd.Tensor,
        kv_cache: Optional[ttml.models.KvCache] = None,
        new_tokens: Optional[int] = None,
    ) -> ttml.autograd.Tensor:
        TILE_SIZE = 32
        input_shape = input.shape()
        actual_seq_len = input_shape[-1]
        padded_seq_len = ((actual_seq_len + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE

        input_padded = input
        if padded_seq_len != actual_seq_len:
            padding = [(0, 0), (0, 0), (0, 0), (0, padded_seq_len - actual_seq_len)]
            input_val_padded = ttnn.pad(input.get_value(), padding=padding, value=0.0)
            input_padded = ttml.autograd.create_tensor(input_val_padded)

        tok_emb_out = self.tok_emb(input_padded)

        out = tok_emb_out
        if padded_seq_len != actual_seq_len:
            slice_start = [0, 0, 0, 0]
            slice_end = [
                tok_emb_out.shape()[0],
                tok_emb_out.shape()[1],
                actual_seq_len,
                tok_emb_out.shape()[3],
            ]
            step = [1, 1, 1, 1]
            out_val = ttnn.slice(tok_emb_out.get_value(), slice_start, slice_end, step)
            out = ttml.autograd.create_tensor(out_val)

        for layer_idx, block in enumerate(self.blocks):
            extra_args = () if kv_cache is None else (kv_cache, layer_idx, new_tokens)
            if self.config.runner_type == ttml.models.RunnerType.MemoryEfficient:
                out = memory_efficient_runner(block, out, mask, *extra_args)
            elif self.config.runner_type == ttml.models.RunnerType.Default:
                out = block(out, mask, *extra_args)
            else:
                raise ValueError(
                    "Unknown runner type. Supported runner types ['default', 'memory_efficient']"
                )

        out = self.ln_fc(out)
        logits = self.fc(out)
        return logits


# C++ Llama bindings from _ttml.models.llama
from ..._ttml.models.llama import (
    CppLlama,
    CppLlamaConfig,
    create_cpp_llama_model,
)

__all__ = [
    # C++ bindings
    "CppLlama",
    "CppLlamaConfig",
    "create_cpp_llama_model",
    # Python implementations
    "Llama",
    "LlamaConfig",
    "LlamaRopeScalingConfig",
]
