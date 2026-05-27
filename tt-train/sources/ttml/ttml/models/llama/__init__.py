# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from math import lcm
from typing import Optional

import ttnn
import ttml
from ttml.modules import (
    AbstractModuleBase,
    ColumnParallelLinear,
    Embedding,
    LinearLayer,
    ModuleList,
)

from .. import RunnerType, WeightTyingType, memory_efficient_runner
from .autograd_ops import SliceLastDim
from .transformer import LlamaBlock, RMSNormLayer, compute_swiglu_intermediate_size


@dataclass(frozen=True)
class LlamaRopeScalingConfig:
    """Llama 3.x RoPE frequency scaling configuration."""

    scaling_factor: float = 0.0  # 0.0 means no scaling
    high_freq_factor: float = 4.0
    low_freq_factor: float = 1.0
    original_context_length: int = 0  # 0 means no scaling


@dataclass(frozen=True)
class LlamaConfig:
    """Llama model hyper-parameters.

    When ``use_tp=True`` the mesh must already be open and the ``"tp"`` axis
    size must evenly divide ``num_attention_heads``, ``num_key_value_heads``,
    and ``intermediate_size`` — this is validated in ``__post_init__``.  The
    vocab does *not* need to be TP-divisible: the embedding and LM-head
    weights are padded internally to ``lcm(32, tp_size)`` and the padded logit
    columns are sliced away before returning.  The effective pre-slice width
    is exposed as ``Llama.padded_vocab_size``.
    """

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
    use_tp: bool = False

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
        if self.use_tp:
            if self.weight_tying == WeightTyingType.Enabled:
                raise ValueError(
                    "weight_tying=Enabled is not supported with use_tp=True: "
                    "tok_emb is replicated but fc is sharded on dim 2, so they "
                    "cannot share a single Parameter."
                )
            tp_size = ttml.mesh().axis_size("tp")
            if self.num_attention_heads % tp_size != 0:
                raise ValueError(
                    "Number of attention heads must be divisible by TP size. "
                    f"num_attention_heads={self.num_attention_heads}, tp_size={tp_size}"
                )
            if self.num_key_value_heads % tp_size != 0:
                raise ValueError(
                    "Number of key/value heads must be divisible by TP size. "
                    f"num_key_value_heads={self.num_key_value_heads}, tp_size={tp_size}"
                )
            intermediate_size = self.intermediate_size
            if intermediate_size is None:
                intermediate_size = compute_swiglu_intermediate_size(self.hidden_size)
            if intermediate_size % tp_size != 0:
                raise ValueError(
                    "Intermediate size must be divisible by TP size. "
                    f"intermediate_size={intermediate_size}, tp_size={tp_size}"
                )


class Llama(AbstractModuleBase):
    """Llama decoder-only transformer (Python implementation)."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()

        self.config = config

        if config.use_tp:
            # Pad the vocab so the LM head's sharded output rows are
            # tile-aligned: ColumnParallelLinear shards dim 2 across TP, so
            # each shard needs to be divisible by 32.  Forward slices the
            # padded logit columns away before returning, so
            # ``config.vocab_size`` is free to be arbitrary.
            tp_size = ttml.mesh().axis_size("tp")
            align = lcm(32, tp_size)
            self.padded_vocab_size = ((config.vocab_size + align - 1) // align) * align
            # gather_output=True: the LM head must produce full-vocab logits
            # on every device so the loss can be computed without further CCL.
            self.fc = ColumnParallelLinear(
                config.hidden_size,
                self.padded_vocab_size,
                has_bias=False,
                gather_output=True,
                axis_name="tp",
            )
        else:
            self.padded_vocab_size = ((config.vocab_size + 31) // 32) * 32
            self.fc = LinearLayer(
                config.hidden_size,
                self.padded_vocab_size,
                False,
            )

        self.tok_emb = Embedding(
            self.padded_vocab_size,
            config.hidden_size,
            weight_init=ttml.init.normal(0.0, 0.02),
        )

        if config.weight_tying == ttml.models.WeightTyingType.Enabled:
            self.tok_emb.weight = self.fc.weight

        head_dim = config.hidden_size // config.num_attention_heads

        rope_scaling_params = ttml.ops.rope.RopeScalingParams()
        if config.rope_scaling.scaling_factor != 0.0 and config.rope_scaling.original_context_length != 0:
            rope_scaling_params.scaling_factor = config.rope_scaling.scaling_factor
            rope_scaling_params.high_freq_factor = config.rope_scaling.high_freq_factor
            rope_scaling_params.low_freq_factor = config.rope_scaling.low_freq_factor
            rope_scaling_params.original_context_length = config.rope_scaling.original_context_length

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
                    use_tp=config.use_tp,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.ln_fc = RMSNormLayer(config.hidden_size)

    def forward(
        self,
        input: ttml.autograd.Tensor,
        mask: ttml.autograd.Tensor,
        kv_cache: Optional[ttml.models.KvCache] = None,
        new_tokens: Optional[int] = None,
    ) -> ttml.autograd.Tensor:
        # Token IDs must be padded to the tile boundary so the embedding lookup
        # produces a tile-aligned tensor.  The padding is stripped after lookup.
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
                raise ValueError("Unknown runner type. Supported runner types ['default', 'memory_efficient']")

        out = self.ln_fc(out)
        logits = self.fc(out)
        if self.padded_vocab_size != self.config.vocab_size:
            logits = SliceLastDim.apply(logits, self.config.vocab_size)
        return logits


# C++ Llama bindings from _ttml.models.llama
from _ttml.models.llama import (
    CppLlama,
    CppLlamaConfig,
    create_cpp_llama_model,
)

from .safetensors_loader import load_from_safetensors
from .flops import calculate_flops_per_token

__all__ = [
    # C++ bindings
    "CppLlama",
    "CppLlamaConfig",
    "create_cpp_llama_model",
    # Python implementations
    "Llama",
    "LlamaConfig",
    "LlamaRopeScalingConfig",
    "calculate_flops_per_token",
    "load_from_safetensors",
]
