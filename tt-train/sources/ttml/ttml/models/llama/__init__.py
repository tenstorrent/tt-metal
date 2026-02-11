# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import ml_dtypes

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, Parameter, ModuleList, RunMode, LinearLayer

from .. import RunnerType
from .embedding import Embedding
from .transformer import LlamaBlock, RMSNormLayer


@dataclass
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
    mlp_dropout: bool = False
    runner_type: RunnerType = RunnerType.Default


def make_llama_config(**kwargs) -> LlamaConfig:
    config = LlamaConfig(**kwargs)
    if config.max_position_embeddings % 32 != 0:
        raise ValueError(
            "Max position embeddings should be divisible by 32 due to current limitations in tensor. "
            f"Provided max_position_embeddings={config.max_position_embeddings}"
        )
    if config.hidden_size % 32 != 0:
        raise ValueError(
            "Hidden size should be divisible by 32 due to current limitations in tensor. "
            f"Provided hidden_size={config.hidden_size}"
        )
    return config


def initialize_parameters(parameters: Dict[str, ttml.autograd.Tensor]) -> None:
    for name, tensor in self.parameters().items():
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

        head_dim = config.hidden_size // config.num_attention_heads
        rope_scaling_params = ttml.ops.rope.RopeScalingParams(
            original_context_length=0,
            scaling_factor=0.0,
            high_freq_factor=4.0,
            low_freq_factor=1.0,
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
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.ln_fc = RMSNormLayer(config.hidden_size)

    def forward(
        self, input: ttml.autograd.Tensor, mask: ttml.autograd.Tensor
    ) -> ttml.autograd.Tensor:
        TILE_SIZE = 32
        input_shape = input.shape()
        actual_seq_len = input_shape[-1]
        padded_seq_len = ((actual_seq_len + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE

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

        out = self.ln_fc(out)
        logits = self.fc(out)
        return logits
