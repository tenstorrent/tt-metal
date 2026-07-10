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
    FeatureParallelEmbedding,
    LinearLayer,
    ModuleList,
    VocabParallelEmbedding,
)

from .. import EmbeddingParallelType, RunnerType, WeightTyingType, memory_efficient_runner
from .autograd_ops import SliceLastDim
from ttml.parallel import TPStrategy
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

    When ``tp_strategy`` enables tensor parallelism the mesh must already be open and the
    ``"tp"`` axis size must evenly divide ``num_attention_heads``, ``num_key_value_heads``,
    and ``intermediate_size`` — this is validated in ``__post_init__``.  The vocab does
    *not* need to be TP-divisible: the embedding and LM-head weights are padded internally
    to ``lcm(32, tp_size)``, exposed as ``Llama.padded_vocab_size``.

    ``embedding_parallel`` selects how the token-embedding table is sharded under TP (see
    :class:`EmbeddingParallelType`); it is ignored when tensor parallelism is disabled.
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
    # Tensor-parallel strategy: NONE / TENSOR / TENSOR_SEQUENCE (the latter adds Megatron
    # sequence parallelism, sharding the residual stream along the sequence across the "tp"
    # axis in the norm/dropout/residual regions).
    tp_strategy: TPStrategy = TPStrategy.NONE
    embedding_parallel: EmbeddingParallelType = EmbeddingParallelType.FeatureParallel

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
        if self.tp_strategy.tensor_parallel:
            if (
                self.embedding_parallel == EmbeddingParallelType.FeatureParallel
                and self.weight_tying == WeightTyingType.Enabled
            ):
                raise ValueError(
                    "embedding_parallel=FeatureParallel shards the token embedding on the "
                    "feature (hidden) dimension, whose layout is incompatible with the "
                    "vocab-parallel LM head; weight tying cannot be used. Set "
                    "weight_tying=Disabled or embedding_parallel=VocabParallel."
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

        if self.tp_strategy.sequence_parallel:
            # Dropout in the sequence-sharded regions would need per-TP-rank RNG
            # (each rank holds different positions); not wired yet, so gate it off.
            if self.attention_dropout > 0.0 or self.mlp_dropout > 0.0:
                raise NotImplementedError(
                    "sequence_parallel does not support dropout>0 yet "
                    f"(attention_dropout={self.attention_dropout}, mlp_dropout={self.mlp_dropout})"
                )
            # Each TP rank owns S/tp_size sequence positions and the sequence
            # reduce-scatter requires the per-shard tile count to divide the ring:
            # (S/32) % tp_size == 0, i.e. S % (32*tp_size) == 0.
            tp_size = ttml.mesh().axis_size("tp")
            if self.max_position_embeddings % (32 * tp_size) != 0:
                raise ValueError(
                    "sequence_parallel requires max_position_embeddings divisible by 32*tp_size "
                    f"(got max_position_embeddings={self.max_position_embeddings}, tp_size={tp_size})"
                )


class Llama(AbstractModuleBase):
    """Llama decoder-only transformer (Python implementation)."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()

        self.config = config

        if config.tp_strategy.tensor_parallel:
            # Pad the vocab so the LM head's sharded output rows are
            # tile-aligned: ColumnParallelLinear shards dim 2 across TP, so
            # each shard needs to be divisible by 32.  The trailing padded
            # columns are kept on-device and handled by the downstream
            # vocab_parallel_cross_entropy_loss, so ``config.vocab_size`` is
            # free to be arbitrary.
            tp_size = ttml.mesh().axis_size("tp")
            align = lcm(32, tp_size)
            self.padded_vocab_size = ((config.vocab_size + align - 1) // align) * align
            # gather_output=False: keep the LM head output vocab-sharded
            # ([B,1,S,padded_V/tp_size] per device) so callers can route through
            # ttml.ops.distributed.vocab_parallel_cross_entropy_loss without an
            # all-gather of the full vocab dimension.
            self.fc = ColumnParallelLinear(
                config.hidden_size,
                self.padded_vocab_size,
                has_bias=False,
                gather_output=False,
                # Under SP the head input (ln_fc output) is sequence-sharded; the
                # column-parallel gather restores the full sequence, yielding
                # full-sequence vocab-sharded logits -- exactly what the classic-TP
                # path produces, so vocab_parallel_cross_entropy_loss is unchanged.
                sequence_parallel=config.tp_strategy.sequence_parallel,
                axis_name="tp",
            )
            if config.embedding_parallel == EmbeddingParallelType.FeatureParallel:
                # Shard the embedding table on the feature (hidden) dim: a fully
                # local lookup plus an all-gather, no id masking. Its layout does
                # not match the vocab-parallel LM head, so weight tying is
                # unavailable (validated in LlamaConfig.__post_init__).
                #
                # Under SP the gathered full-hidden embedding is additionally
                # scattered along the sequence, so the first block receives a
                # sequence-sharded residual.
                self.tok_emb = FeatureParallelEmbedding(
                    self.padded_vocab_size,
                    config.hidden_size,
                    weight_init=ttml.init.normal(0.0, 0.02),
                    sequence_parallel=config.tp_strategy.sequence_parallel,
                    axis_name="tp",
                )
            else:
                # Shard the embedding table on the vocab dim to mirror the LM head:
                # each device keeps only its vocab slice instead of a full replicated
                # table, and the matching layout allows a tied weight (below).
                #
                # Under SP the embedding output is reduce-scattered along the
                # sequence so the first block receives a sequence-sharded residual.
                self.tok_emb = VocabParallelEmbedding(
                    self.padded_vocab_size,
                    config.hidden_size,
                    weight_init=ttml.init.normal(0.0, 0.02),
                    sequence_parallel=config.tp_strategy.sequence_parallel,
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
                    tp_strategy=config.tp_strategy,
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

        if self.config.tp_strategy.sequence_parallel:
            # SP shards the embedding output along the sequence, so the post-embedding
            # unpad-slice (on the now-sharded sequence dim) is not expressible. Require
            # the sequence to already be tile*tp aligned so no pad/slice is needed; the
            # sequence reduce-scatter needs the same alignment anyway.
            tp_size = ttml.mesh().axis_size("tp")
            if actual_seq_len % (TILE_SIZE * tp_size) != 0:
                raise ValueError(
                    "sequence_parallel requires the input sequence length divisible by 32*tp_size "
                    f"(got seq_len={actual_seq_len}, tp_size={tp_size})"
                )

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
        # In TP mode the LM head output stays vocab-sharded; the trailing
        # padded columns are handled by vocab_parallel_cross_entropy_loss.
        # The non-TP path returns full-vocab logits, so we still need to drop
        # the tile-alignment padding before handing them off to the caller.
        if not self.config.tp_strategy.tensor_parallel and self.padded_vocab_size != self.config.vocab_size:
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
