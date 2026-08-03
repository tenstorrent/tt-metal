# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import torch

import ttnn

from .config import TimeSeriesTransformerConfig
from .ops import linear, pad_last_dim, softmax
from .weights import LoadResult, load_tensors, upload


@dataclass
class KeyValueCache:
    """Projected keys and values held across autoregressive decode steps.

    ``key``/``value`` are head-split, shape ``(batch, heads, length, head_dim)``. Cross-
    attention caches are filled once from the encoder output and never grow; self-attention
    caches grow by one timestep per decode step.
    """

    key: Optional[ttnn.Tensor] = None
    value: Optional[ttnn.Tensor] = None

    @property
    def length(self) -> int:
        return 0 if self.key is None else int(self.key.shape[2])

    @property
    def is_filled(self) -> bool:
        return self.key is not None

    def append(self, key: ttnn.Tensor, value: ttnn.Tensor) -> None:
        if self.key is None:
            self.key, self.value = key, value
        else:
            self.key = ttnn.concat([self.key, key], dim=2)
            self.value = ttnn.concat([self.value, value], dim=2)

    def reset(self) -> None:
        self.key = None
        self.value = None


class MultiHeadAttention:
    """``TimeSeriesTransformerAttention``: standard scaled dot-product multi-head attention.

    Serves encoder self-attention, decoder masked self-attention, and decoder cross-attention;
    the three differ only in where keys/values come from and whether a mask is supplied.
    """

    def __init__(
        self,
        config: TimeSeriesTransformerConfig,
        *,
        device,
        dtype: ttnn.DataType,
        memory_config: Optional[ttnn.MemoryConfig] = None,
        rng: Optional[torch.Generator] = None,
    ):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.memory_config = memory_config
        self.num_heads = config.encoder_attention_heads
        self.head_dim = config.head_dim
        self.scaling = float(self.head_dim**-0.5)
        self.use_sdpa = config.use_sdpa
        self.use_exact_softmax = config.use_exact_softmax
        # Diagnostic hook for the PCC tests; holds the last softmax output.
        self.last_attention_probs: Optional[ttnn.Tensor] = None

        d_model = config.d_model
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            if rng is None:
                weight = torch.zeros((d_model, d_model), dtype=torch.float32)
            else:
                weight = torch.randn((d_model, d_model), generator=rng, dtype=torch.float32) * 0.02
            bias = torch.zeros((d_model,), dtype=torch.float32)
            setattr(self, f"{name}_weight_torch", weight)
            setattr(self, f"{name}_bias_torch", bias)
            setattr(self, f"{name}_weight", upload(weight, device=device, dtype=dtype))
            setattr(self, f"{name}_bias", upload(bias, device=device, dtype=dtype))

    def load_hf_state_dict(self, state: Mapping[str, torch.Tensor], *, strict: bool = True) -> LoadResult:
        mapping = tuple(
            (f"{name}.{suffix}", f"{name}_{attr}")
            for name in ("q_proj", "k_proj", "v_proj", "out_proj")
            for suffix, attr in (("weight", "weight"), ("bias", "bias"))
        )
        return load_tensors(
            self,
            state,
            mapping,
            device=self.device,
            dtype=self.dtype,
            strict=strict,
            label="attention",
        )

    # -- shape plumbing -----------------------------------------------------

    def split_heads(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """``(batch, seq, d_model)`` -> ``(batch, heads, seq, head_dim)``."""
        batch, seq = int(x.shape[0]), int(x.shape[1])
        x = ttnn.reshape(x, (batch, seq, self.num_heads, self.head_dim))
        return ttnn.permute(x, (0, 2, 1, 3))

    def merge_heads(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """``(batch, heads, seq, head_dim)`` -> ``(batch, seq, d_model)``."""
        batch, seq = int(x.shape[0]), int(x.shape[2])
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.reshape(x, (batch, seq, self.num_heads * self.head_dim))

    def project(self, x: ttnn.Tensor, name: str) -> ttnn.Tensor:
        return linear(
            x,
            getattr(self, f"{name}_weight"),
            getattr(self, f"{name}_bias"),
            dtype=self.dtype,
            memory_config=self.memory_config,
        )

    # -- attention cores ----------------------------------------------------

    def eager_attention(
        self,
        query: ttnn.Tensor,
        key: ttnn.Tensor,
        value: ttnn.Tensor,
        mask: Optional[ttnn.Tensor],
    ) -> ttnn.Tensor:
        scores = ttnn.matmul(query, ttnn.permute(key, (0, 1, 3, 2)), dtype=self.dtype)
        scores = ttnn.multiply(scores, self.scaling)
        if mask is not None:
            scores = ttnn.add(scores, mask)
        probs = softmax(scores, dim=-1, exact=self.use_exact_softmax)
        self.last_attention_probs = probs
        return ttnn.matmul(probs, value, dtype=self.dtype)

    def sdpa_attention(
        self,
        query: ttnn.Tensor,
        key: ttnn.Tensor,
        value: ttnn.Tensor,
        *,
        is_causal: bool,
    ) -> ttnn.Tensor:
        """Flash-attention path.

        The kernel rejects a logical last dim that differs from the tile-padded one, so
        head_dim=13 must be zero-padded to 32 first. Padding with zeros leaves QK^T and the
        context matmul exact, since the extra columns contribute nothing to either.
        """
        padded_query, width = pad_last_dim(query)
        padded_key, _ = pad_last_dim(key)
        padded_value, _ = pad_last_dim(value)
        output = ttnn.transformer.scaled_dot_product_attention(
            padded_query, padded_key, padded_value, is_causal=is_causal, scale=self.scaling
        )
        self.last_attention_probs = None
        if int(output.shape[-1]) == width:
            return output
        starts = [0] * len(output.shape)
        ends = list(output.shape)
        ends[-1] = width
        return ttnn.slice(output, starts, ends)

    # -- entry point --------------------------------------------------------

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        key_value_states: Optional[ttnn.Tensor] = None,
        *,
        attention_mask: Optional[ttnn.Tensor] = None,
        cache: Optional[KeyValueCache] = None,
        is_causal: bool = False,
    ) -> ttnn.Tensor:
        """Attend ``hidden_states`` over ``key_value_states`` (defaults to self-attention).

        When ``cache`` is supplied: a cross-attention cache is filled on first use and reused
        verbatim afterwards; a self-attention cache is extended with this step's keys/values.
        """
        is_cross_attention = key_value_states is not None
        query = self.split_heads(self.project(hidden_states, "q_proj"))

        if cache is not None and is_cross_attention and cache.is_filled:
            key, value = cache.key, cache.value
        else:
            source = key_value_states if is_cross_attention else hidden_states
            key = self.split_heads(self.project(source, "k_proj"))
            value = self.split_heads(self.project(source, "v_proj"))
            if cache is not None:
                if is_cross_attention:
                    cache.key, cache.value = key, value
                else:
                    cache.append(key, value)
                    key, value = cache.key, cache.value

        if self.use_sdpa and attention_mask is None:
            context = self.sdpa_attention(query, key, value, is_causal=is_causal)
        else:
            context = self.eager_attention(query, key, value, attention_mask)

        return self.project(self.merge_heads(context), "out_proj")


__all__ = ["KeyValueCache", "MultiHeadAttention"]
