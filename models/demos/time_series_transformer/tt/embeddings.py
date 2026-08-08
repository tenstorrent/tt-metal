# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Optional

import torch

import ttnn

from .config import TimeSeriesTransformerConfig
from .layers import LayerNorm, Linear
from .weights import LoadResult, to_float_tensor, upload


def sinusoidal_position_encoding(length: int, d_model: int) -> torch.Tensor:
    """Reproduce ``TimeSeriesSinusoidalPositionalEmbedding``: a sin block then a cos block.

    Note this is *not* the interleaved sin/cos of the original Transformer paper.
    """
    position = torch.arange(length, dtype=torch.float32)[:, None]
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    sin = torch.sin(position * div_term)
    cos = torch.cos(position * div_term)
    sentinel = d_model // 2 if d_model % 2 == 0 else (d_model // 2) + 1
    encoding = torch.zeros((length, d_model), dtype=torch.float32)
    encoding[:, 0:sentinel] = sin
    encoding[:, sentinel:] = cos[:, : d_model - sentinel]
    return encoding


class SinusoidalPositionalEmbedding:
    """Fixed positional table, sliced by ``(offset, length)``.

    Slices are cut on the host and cached per window rather than sliced on device: the table
    is tiny, and a device-side slice of a tile-laid-out tensor at a non-tile-aligned row
    (the decoder starts at position ``context_length=24``) is not free.
    """

    def __init__(self, max_positions: int, d_model: int, *, device, dtype: ttnn.DataType):
        self.device = device
        self.dtype = dtype
        self.max_positions = max_positions
        self.weight_torch = sinusoidal_position_encoding(max_positions, d_model)
        self._cache: dict[tuple[int, int], ttnn.Tensor] = {}

    def load_hf_state_dict(self, state: Mapping[str, torch.Tensor], *, strict: bool = True) -> LoadResult:
        weight = state.get("weight")
        if weight is None:
            if strict:
                raise ValueError("Missing positional embedding weight.")
            return {"missing_keys": ["weight"], "unexpected_keys": sorted(state)}
        self.weight_torch = to_float_tensor(weight)
        self.max_positions = int(self.weight_torch.shape[0])
        self._cache.clear()
        return {"missing_keys": [], "unexpected_keys": sorted(k for k in state if k != "weight")}

    def __call__(self, length: int, *, offset: int = 0) -> ttnn.Tensor:
        if offset + length > self.max_positions:
            raise ValueError(
                f"Positional slice [{offset}:{offset + length}] exceeds table of {self.max_positions} positions."
            )
        key = (offset, length)
        cached = self._cache.get(key)
        if cached is None:
            window = self.weight_torch[offset : offset + length].reshape(1, length, -1)
            cached = upload(window, device=self.device, dtype=self.dtype)
            self._cache[key] = cached
        return cached


class TimeSeriesEmbedding:
    """Encoder/decoder input block: ``layernorm(value_embedding(x) + positions)``."""

    def __init__(
        self,
        config: TimeSeriesTransformerConfig,
        *,
        device,
        dtype: ttnn.DataType,
        memory_config: Optional[ttnn.MemoryConfig] = None,
        rng: Optional[torch.Generator] = None,
    ):
        assert config.feature_size is not None
        self.config = config
        # HF's TimeSeriesValueEmbedding is a bias-free projection from feature_size to d_model.
        self.value_embedding = Linear(
            config.d_model,
            config.feature_size,
            device=device,
            dtype=dtype,
            use_bias=False,
            memory_config=memory_config,
            rng=rng,
        )
        self.embed_positions = SinusoidalPositionalEmbedding(
            config.max_position_embeddings, config.d_model, device=device, dtype=dtype
        )
        self.layernorm_embedding = LayerNorm(config.d_model, device=device, dtype=dtype, eps=config.layer_norm_eps)

    def load_hf_state_dict(self, state: Mapping[str, torch.Tensor], *, strict: bool = True) -> LoadResult:
        from .weights import merge_results, substate

        return merge_results(
            [
                (
                    "value_embedding",
                    self.value_embedding.load_hf_state_dict(
                        substate(state, "value_embedding.value_projection"), strict=strict
                    ),
                ),
                (
                    "embed_positions",
                    self.embed_positions.load_hf_state_dict(substate(state, "embed_positions"), strict=strict),
                ),
                (
                    "layernorm_embedding",
                    self.layernorm_embedding.load_hf_state_dict(substate(state, "layernorm_embedding"), strict=strict),
                ),
            ],
            state=state,
            claimed=("value_embedding", "embed_positions", "layernorm_embedding"),
        )

    def __call__(self, x: ttnn.Tensor, *, position_offset: int = 0) -> ttnn.Tensor:
        hidden = self.value_embedding(x)
        positions = self.embed_positions(int(hidden.shape[1]), offset=position_offset)
        return self.layernorm_embedding(ttnn.add(hidden, positions))


__all__ = [
    "SinusoidalPositionalEmbedding",
    "TimeSeriesEmbedding",
    "sinusoidal_position_encoding",
]
