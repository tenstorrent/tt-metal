# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional

import torch

import ttnn

from .attention import KeyValueCache, MultiHeadAttention
from .config import TimeSeriesTransformerConfig, get_memory_config
from .embeddings import TimeSeriesEmbedding
from .layers import FeedForward, LayerNorm
from .ops import make_causal_mask, make_causal_mask_with_offset
from .weights import LoadResult, merge_results, pick_roots, substate


@dataclass
class LayerCache:
    """Per-decoder-layer caches: growing self-attention, fixed cross-attention."""

    self_attention: KeyValueCache = field(default_factory=KeyValueCache)
    cross_attention: KeyValueCache = field(default_factory=KeyValueCache)

    def reset(self) -> None:
        self.self_attention.reset()
        self.cross_attention.reset()


class EncoderLayer:
    """Post-norm encoder block: attention -> add & norm -> FFN -> add & norm."""

    def __init__(
        self,
        config: TimeSeriesTransformerConfig,
        *,
        device,
        dtype: ttnn.DataType,
        rng: Optional[torch.Generator] = None,
    ):
        memory_config = get_memory_config(config)
        self.self_attn = MultiHeadAttention(config, device=device, dtype=dtype, memory_config=memory_config, rng=rng)
        self.self_attn_layer_norm = LayerNorm(config.d_model, device=device, dtype=dtype, eps=config.layer_norm_eps)
        self.ffn = FeedForward(
            config.d_model,
            config.encoder_ffn_dim,
            device=device,
            dtype=dtype,
            activation_function=config.activation_function,
            memory_config=memory_config,
            rng=rng,
        )
        self.final_layer_norm = LayerNorm(config.d_model, device=device, dtype=dtype, eps=config.layer_norm_eps)

    def load_hf_state_dict(self, state: Mapping[str, torch.Tensor], *, strict: bool = True) -> LoadResult:
        return merge_results(
            [
                ("self_attn", self.self_attn.load_hf_state_dict(substate(state, "self_attn"), strict=strict)),
                (
                    "self_attn_layer_norm",
                    self.self_attn_layer_norm.load_hf_state_dict(
                        substate(state, "self_attn_layer_norm"), strict=strict
                    ),
                ),
                ("", self.ffn.load_hf_state_dict(pick_roots(state, ("fc1", "fc2")), strict=strict)),
                (
                    "final_layer_norm",
                    self.final_layer_norm.load_hf_state_dict(substate(state, "final_layer_norm"), strict=strict),
                ),
            ],
            state=state,
            claimed=("self_attn", "self_attn_layer_norm", "fc1", "fc2", "final_layer_norm"),
        )

    def __call__(self, hidden_states: ttnn.Tensor, attention_mask: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        attended = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = self.self_attn_layer_norm(ttnn.add(hidden_states, attended))
        hidden_states = self.final_layer_norm(ttnn.add(hidden_states, self.ffn(hidden_states)))
        return hidden_states


class DecoderLayer:
    """Post-norm decoder block: masked self-attention, cross-attention, FFN."""

    def __init__(
        self,
        config: TimeSeriesTransformerConfig,
        *,
        device,
        dtype: ttnn.DataType,
        rng: Optional[torch.Generator] = None,
    ):
        memory_config = get_memory_config(config)
        self.self_attn = MultiHeadAttention(config, device=device, dtype=dtype, memory_config=memory_config, rng=rng)
        self.self_attn_layer_norm = LayerNorm(config.d_model, device=device, dtype=dtype, eps=config.layer_norm_eps)
        self.encoder_attn = MultiHeadAttention(config, device=device, dtype=dtype, memory_config=memory_config, rng=rng)
        self.encoder_attn_layer_norm = LayerNorm(config.d_model, device=device, dtype=dtype, eps=config.layer_norm_eps)
        self.ffn = FeedForward(
            config.d_model,
            config.decoder_ffn_dim,
            device=device,
            dtype=dtype,
            activation_function=config.activation_function,
            memory_config=memory_config,
            rng=rng,
        )
        self.final_layer_norm = LayerNorm(config.d_model, device=device, dtype=dtype, eps=config.layer_norm_eps)

    def load_hf_state_dict(self, state: Mapping[str, torch.Tensor], *, strict: bool = True) -> LoadResult:
        return merge_results(
            [
                ("self_attn", self.self_attn.load_hf_state_dict(substate(state, "self_attn"), strict=strict)),
                (
                    "self_attn_layer_norm",
                    self.self_attn_layer_norm.load_hf_state_dict(
                        substate(state, "self_attn_layer_norm"), strict=strict
                    ),
                ),
                (
                    "encoder_attn",
                    self.encoder_attn.load_hf_state_dict(substate(state, "encoder_attn"), strict=strict),
                ),
                (
                    "encoder_attn_layer_norm",
                    self.encoder_attn_layer_norm.load_hf_state_dict(
                        substate(state, "encoder_attn_layer_norm"), strict=strict
                    ),
                ),
                ("", self.ffn.load_hf_state_dict(pick_roots(state, ("fc1", "fc2")), strict=strict)),
                (
                    "final_layer_norm",
                    self.final_layer_norm.load_hf_state_dict(substate(state, "final_layer_norm"), strict=strict),
                ),
            ],
            state=state,
            claimed=(
                "self_attn",
                "self_attn_layer_norm",
                "encoder_attn",
                "encoder_attn_layer_norm",
                "fc1",
                "fc2",
                "final_layer_norm",
            ),
        )

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        *,
        attention_mask: Optional[ttnn.Tensor] = None,
        cache: Optional[LayerCache] = None,
        cross_cache: Optional[KeyValueCache] = None,
        is_causal: bool = False,
    ) -> ttnn.Tensor:
        """Decode ``hidden_states``.

        ``cross_cache`` reuses cross-attention keys and values without also caching
        self-attention, which the full-window rollout needs: the encoder output is constant
        across a forecast, so its projections are recomputed 24 times for nothing otherwise.
        """
        attended = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            cache=None if cache is None else cache.self_attention,
            is_causal=is_causal,
        )
        hidden_states = self.self_attn_layer_norm(ttnn.add(hidden_states, attended))

        crossed = self.encoder_attn(
            hidden_states,
            encoder_hidden_states,
            cache=cache.cross_attention if cache is not None else cross_cache,
        )
        hidden_states = self.encoder_attn_layer_norm(ttnn.add(hidden_states, crossed))

        hidden_states = self.final_layer_norm(ttnn.add(hidden_states, self.ffn(hidden_states)))
        return hidden_states


class Encoder:
    """Embedding block plus a stack of encoder layers."""

    def __init__(
        self,
        config: TimeSeriesTransformerConfig,
        *,
        device,
        dtype: ttnn.DataType,
        rng: Optional[torch.Generator] = None,
    ):
        self.config = config
        self.embedding = TimeSeriesEmbedding(
            config, device=device, dtype=dtype, memory_config=get_memory_config(config), rng=rng
        )
        self.layers = [EncoderLayer(config, device=device, dtype=dtype, rng=rng) for _ in range(config.encoder_layers)]

    def load_hf_state_dict(self, state: Mapping[str, torch.Tensor], *, strict: bool = True) -> LoadResult:
        embedding_roots = ("value_embedding", "embed_positions", "layernorm_embedding")
        results = [("", self.embedding.load_hf_state_dict(pick_roots(state, embedding_roots), strict=strict))]
        for index, layer in enumerate(self.layers):
            prefix = f"layers.{index}"
            results.append((prefix, layer.load_hf_state_dict(substate(state, prefix), strict=strict)))
        return merge_results(results, state=state, claimed=embedding_roots + ("layers",))

    def __call__(self, inputs: ttnn.Tensor, *, output_hidden_states: bool = False):
        hidden_states = self.embedding(inputs, position_offset=0)
        collected = [hidden_states] if output_hidden_states else None
        for layer in self.layers:
            # Every encoder position is valid, so no padding mask is required.
            hidden_states = layer(hidden_states)
            if collected is not None:
                collected.append(hidden_states)
        return (hidden_states, collected) if output_hidden_states else hidden_states


class Decoder:
    """Embedding block plus a stack of decoder layers, with optional KV caching."""

    def __init__(
        self,
        config: TimeSeriesTransformerConfig,
        *,
        device,
        dtype: ttnn.DataType,
        rng: Optional[torch.Generator] = None,
    ):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.embedding = TimeSeriesEmbedding(
            config, device=device, dtype=dtype, memory_config=get_memory_config(config), rng=rng
        )
        self.layers = [DecoderLayer(config, device=device, dtype=dtype, rng=rng) for _ in range(config.decoder_layers)]

    def load_hf_state_dict(self, state: Mapping[str, torch.Tensor], *, strict: bool = True) -> LoadResult:
        embedding_roots = ("value_embedding", "embed_positions", "layernorm_embedding")
        results = [("", self.embedding.load_hf_state_dict(pick_roots(state, embedding_roots), strict=strict))]
        for index, layer in enumerate(self.layers):
            prefix = f"layers.{index}"
            results.append((prefix, layer.load_hf_state_dict(substate(state, prefix), strict=strict)))
        return merge_results(results, state=state, claimed=embedding_roots + ("layers",))

    def new_caches(self) -> list[LayerCache]:
        return [LayerCache() for _ in self.layers]

    def new_cross_caches(self) -> list[KeyValueCache]:
        """Cross-attention caches only, for callers that recompute self-attention each step."""
        return [KeyValueCache() for _ in self.layers]

    def __call__(
        self,
        inputs: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        *,
        position_offset: Optional[int] = None,
        caches: Optional[list[LayerCache]] = None,
        cross_caches: Optional[list[KeyValueCache]] = None,
        output_hidden_states: bool = False,
    ):
        """Decode ``inputs``, which start at ``position_offset`` in the positional table.

        HF places decoder positions after the encoder's in a shared table, so the default
        offset is ``context_length`` plus however many steps are already cached.
        """
        cached_length = caches[0].self_attention.length if caches else 0
        if position_offset is None:
            position_offset = self.config.context_length + cached_length

        hidden_states = self.embedding(inputs, position_offset=position_offset)
        query_length = int(hidden_states.shape[1])

        is_causal = False
        if caches is None:
            if self.config.use_sdpa:
                # The flash kernel applies causality itself; an additive mask would send it
                # down the eager path and defeat the point.
                mask = None
                is_causal = True
            else:
                mask = make_causal_mask(
                    query_length,
                    device=self.device,
                    dtype=self.dtype,
                    mask_value=self.config.attn_mask_value,
                )
        else:
            # With a cache every stored key precedes the current query block, so a single-token
            # step needs no mask at all (make_causal_mask_with_offset returns None).
            mask = make_causal_mask_with_offset(
                query_length,
                cached_length + query_length,
                device=self.device,
                dtype=self.dtype,
                mask_value=self.config.attn_mask_value,
            )

        collected = [hidden_states] if output_hidden_states else None
        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states,
                attention_mask=mask,
                cache=caches[index] if caches else None,
                cross_cache=cross_caches[index] if cross_caches else None,
                is_causal=is_causal,
            )
            if collected is not None:
                collected.append(hidden_states)
        return (hidden_states, collected) if output_hidden_states else hidden_states


__all__ = [
    "Decoder",
    "DecoderLayer",
    "Encoder",
    "EncoderLayer",
    "LayerCache",
]
