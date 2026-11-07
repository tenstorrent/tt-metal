# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-resident exact PLE lookup and post-lookup numerical oracle."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.reference import (
    NGramHashSpec,
    build_ngram_hash_spec,
    causal_depthwise_conv1d,
    ngram_token_ids,
    zero_centered_rms_norm,
)

TP_SIZE = 4


class Qwen38HostPLEEmbedding:
    """Synchronous sparse lookup over the 128 on-disk host table parts."""

    def __init__(self, checkpoint: Qwen38Checkpoint):
        self.checkpoint = checkpoint
        self.config = checkpoint.config
        prefix = (
            f"model.language_model.layers.{self.config.ple_checkpoint_layer}.ple."
            "ple_embedding.ngram_embedding.shard_"
        )
        pattern = re.compile(re.escape(prefix) + r"([0-9]+)\.weight")
        indexed_names: dict[int, str] = {}
        for name in checkpoint.names_with_prefix(prefix):
            match = pattern.fullmatch(name)
            if match is None:
                raise ValueError(f"unexpected PLE table tensor name: {name}")
            indexed_names[int(match.group(1))] = name
        if tuple(sorted(indexed_names)) != tuple(range(self.config.ngram_shards)):
            raise ValueError("PLE table parts are not exactly the numeric shard range 0..127")
        self.table_names = tuple(indexed_names[index] for index in range(self.config.ngram_shards))
        shapes = {checkpoint.metadata(name).shape for name in self.table_names}
        dtypes = {checkpoint.metadata(name).dtype for name in self.table_names}
        if len(shapes) != 1 or dtypes != {"BF16"}:
            raise ValueError(f"PLE table parts disagree in shape/dtype: shapes={shapes}, dtypes={dtypes}")
        self.rows_per_shard, self.embedding_head_dim = next(iter(shapes))
        self.ngram_heads = (self.config.ngram_size - 1) * self.config.heads_per_ngram
        if self.ngram_heads * self.embedding_head_dim != self.config.ple_embedding_width:
            raise ValueError("PLE per-head table width does not reconstruct ple_embed_dim")

        metadata_prefix = f"model.language_model.layers.{self.config.ple_checkpoint_layer}.ple.ple_embedding."
        multipliers = checkpoint.tensor(metadata_prefix + "layer_multipliers").long()
        vocab_sizes = checkpoint.tensor(metadata_prefix + "ngram_heads_vocab_sizes").long()
        offsets = checkpoint.tensor(metadata_prefix + "ngram_heads_offsets").long()
        computed = build_ngram_hash_spec(
            unigram_vocab_size=self.config.vocab_size,
            ngram_size=self.config.ngram_size,
            heads_per_ngram=self.config.heads_per_ngram,
            ngram_vocab_size_base=self.config.ngram_vocab_size_base,
            ple_layer_index=0,
            seed=self.config.ple_seed,
            divisible_by=self.config.ngram_divisible_by,
        )
        if not torch.equal(multipliers, computed.layer_multipliers):
            raise ValueError("checkpoint PLE layer multipliers disagree with the pinned hash algorithm")
        if not torch.equal(vocab_sizes, computed.head_vocab_sizes) or not torch.equal(offsets, computed.head_offsets):
            raise ValueError("checkpoint PLE vocab sizes/offsets disagree with the pinned prime sequence")
        padded_rows = self.rows_per_shard * len(self.table_names)
        if padded_rows != computed.padded_vocab_size:
            raise ValueError(
                f"PLE table rows {padded_rows} disagree with padded vocabulary {computed.padded_vocab_size}"
            )
        self.spec = NGramHashSpec(
            ngram_size=computed.ngram_size,
            heads_per_ngram=computed.heads_per_ngram,
            layer_multipliers=multipliers,
            head_vocab_sizes=vocab_sizes,
            head_offsets=offsets,
            padded_vocab_size=padded_rows,
        )

    def lookup(
        self, input_ids: torch.Tensor, previous_context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.device.type != "cpu":
            raise ValueError("PLE hash and table lookup are host-resident")
        ids, next_context = ngram_token_ids(input_ids, previous_context, self.config.eos_token_id, self.spec)
        flat_ids = ids.contiguous().view(-1)
        output = torch.empty((flat_ids.numel(), self.embedding_head_dim), dtype=torch.bfloat16)
        shard_indices = torch.div(flat_ids, self.rows_per_shard, rounding_mode="floor")
        for shard_index in torch.unique(shard_indices, sorted=True).tolist():
            positions = torch.nonzero(shard_indices == shard_index, as_tuple=False).flatten()
            local_rows = flat_ids.index_select(0, positions) - shard_index * self.rows_per_shard
            rows = self.checkpoint.tensor_rows(self.table_names[shard_index], local_rows)
            output.index_copy_(0, positions, rows)
        embeddings = output.reshape(*ids.shape, self.embedding_head_dim).flatten(-2)
        return embeddings, next_context

    @staticmethod
    def shard_result(embeddings: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if embeddings.shape[-1] != 2560:
            raise ValueError(f"PLE embeddings must have width 2560, got {embeddings.shape[-1]}")
        return tuple(part.contiguous() for part in embeddings.chunk(TP_SIZE, dim=-1))


@dataclass(frozen=True)
class Qwen38PLEWeights:
    layer_idx: int
    hidden_size: int
    residual_branches: int
    embedding_width: int
    conv_kernel: int
    conv_dilation: int
    rms_norm_eps: float
    key: torch.Tensor
    value: torch.Tensor
    norm_key: torch.Tensor
    norm_query: torch.Tensor
    norm_conv: torch.Tensor
    conv: torch.Tensor

    @classmethod
    def from_checkpoint(cls, checkpoint: Qwen38Checkpoint) -> "Qwen38PLEWeights":
        config = checkpoint.config
        layer_idx = config.ple_checkpoint_layer
        prefix = f"model.language_model.layers.{layer_idx}.ple."
        tensors = {
            field: checkpoint.tensor(prefix + checkpoint_name)
            for field, checkpoint_name in {
                "key": "key_proj.weight",
                "value": "value_proj.weight",
                "norm_key": "norm_key.weight",
                "norm_query": "norm_query.weight",
                "norm_conv": "norm_conv.weight",
                "conv": "conv1d.weight",
            }.items()
        }
        residual_width = config.residual_width
        expected_shapes = {
            "key": (residual_width, config.ple_embedding_width),
            "value": (config.hidden_size, config.ple_embedding_width),
            "norm_key": (residual_width,),
            "norm_query": (residual_width,),
            "norm_conv": (residual_width,),
            "conv": (residual_width, 1, config.ple_conv_kernel),
        }
        for name, expected in expected_shapes.items():
            tensor = tensors[name]
            if tuple(tensor.shape) != expected:
                raise ValueError(f"PLE {name} shape must be {expected}, got {tuple(tensor.shape)}")
            if tensor.dtype != torch.bfloat16:
                raise ValueError(f"PLE {name} must be BF16, got {tensor.dtype}")
        return cls(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            residual_branches=config.residual_branches,
            embedding_width=config.ple_embedding_width,
            conv_kernel=config.ple_conv_kernel,
            conv_dilation=config.ngram_size,
            rms_norm_eps=config.rms_norm_eps,
            **tensors,
        )

    def transformers_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "key_proj.weight": self.key,
            "value_proj.weight": self.value,
            "norm_key.weight": self.norm_key,
            "norm_query.weight": self.norm_query,
            "norm_conv.weight": self.norm_conv,
            "conv1d.weight": self.conv,
        }


@dataclass(frozen=True)
class Qwen38PLEState:
    token_context: torch.Tensor
    conv: torch.Tensor


class Qwen38PLE:
    """Exact post-lookup PLE equations with explicit token/conv state."""

    def __init__(self, embedding: Qwen38HostPLEEmbedding, weights: Qwen38PLEWeights):
        if embedding.config.ple_checkpoint_layer != weights.layer_idx:
            raise ValueError("PLE embedding and projection weights come from different layers")
        self.embedding = embedding
        self.weights = weights

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor, state: Qwen38PLEState | None = None
    ) -> tuple[torch.Tensor, Qwen38PLEState]:
        weights = self.weights
        residual_width = weights.residual_branches * weights.hidden_size
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != residual_width:
            raise ValueError(f"PLE hidden_states must be [batch, sequence, {residual_width}]")
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(f"PLE oracle requires BF16 activations, got {hidden_states.dtype}")
        if input_ids.shape != hidden_states.shape[:2] or input_ids.dtype != torch.long:
            raise ValueError("PLE input_ids must be int64 [batch, sequence] aligned with hidden_states")
        batch = hidden_states.shape[0]
        conv_state_len = (weights.conv_kernel - 1) * weights.conv_dilation
        if state is not None:
            if tuple(state.token_context.shape) != (batch, self.embedding.config.ngram_size - 1):
                raise ValueError("PLE token-history state has the wrong shape")
            if tuple(state.conv.shape) != (batch, residual_width, conv_state_len):
                raise ValueError("PLE dilated-convolution state has the wrong shape")

        embeddings, next_context = self.embedding.lookup(input_ids, None if state is None else state.token_context)
        key = F.linear(embeddings, weights.key)
        key = zero_centered_rms_norm(key, weights.norm_key, weights.rms_norm_eps, group_size=weights.hidden_size)
        key = key.unflatten(-1, (weights.residual_branches, weights.hidden_size))
        value = F.linear(embeddings, weights.value)
        query = zero_centered_rms_norm(
            hidden_states, weights.norm_query, weights.rms_norm_eps, group_size=weights.hidden_size
        ).unflatten(-1, (weights.residual_branches, weights.hidden_size))
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(weights.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value = gated_value.flatten(-2)
        normalized = zero_centered_rms_norm(
            gated_value, weights.norm_conv, weights.rms_norm_eps, group_size=weights.hidden_size
        )
        convolution, next_conv = causal_depthwise_conv1d(
            normalized.transpose(1, 2),
            weights.conv.squeeze(1),
            None if state is None else state.conv,
            dilation=weights.conv_dilation,
            activation="silu",
        )
        output = gated_value + convolution.transpose(1, 2)
        return output, Qwen38PLEState(token_context=next_context, conv=next_conv)
