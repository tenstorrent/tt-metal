# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import torch

import ttnn

from .ops import activation, linear
from .weights import LoadResult, load_tensors, to_float_tensor, upload


class LayerNorm:
    """Elementwise-affine layer norm over the last dimension.

    The reference checkpoint normalizes over d_model=26, which tile-pads to 32. TTNN reduces
    over the logical width, so no masking is needed here (verified on Wormhole: exact match
    against ``torch.nn.functional.layer_norm``).
    """

    def __init__(self, dim: int, *, device, dtype: ttnn.DataType, eps: float = 1e-5):
        self.device = device
        self.dtype = dtype
        self.eps = eps
        self.weight_torch = torch.ones((dim,), dtype=torch.float32)
        self.bias_torch = torch.zeros((dim,), dtype=torch.float32)
        self.weight = upload(self.weight_torch.reshape(1, -1), device=device, dtype=dtype)
        self.bias = upload(self.bias_torch.reshape(1, -1), device=device, dtype=dtype)

    def load_hf_state_dict(self, state: Mapping[str, torch.Tensor], *, strict: bool = True) -> LoadResult:
        used: set[str] = set()
        missing: list[str] = []
        for key in ("weight", "bias"):
            tensor = state.get(key)
            if tensor is None:
                missing.append(key)
                continue
            used.add(key)
            value = to_float_tensor(tensor)
            setattr(self, f"{key}_torch", value)
            # ttnn.layer_norm expects the affine terms as a row vector.
            setattr(self, key, upload(value.reshape(1, -1), device=self.device, dtype=self.dtype))
        if strict and missing:
            raise ValueError(f"Missing layer norm weights: {missing}")
        return {"missing_keys": missing, "unexpected_keys": sorted(k for k in state if k not in used)}

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(x, weight=self.weight, bias=self.bias, epsilon=self.eps)


class FeedForward:
    """Position-wise FFN: ``fc2(act(fc1(x)))``."""

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        *,
        device,
        dtype: ttnn.DataType,
        activation_function: str = "gelu",
        memory_config: Optional[ttnn.MemoryConfig] = None,
        rng: Optional[torch.Generator] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.activation_function = activation_function
        self.memory_config = memory_config

        def init(shape: tuple[int, ...]) -> torch.Tensor:
            if rng is None:
                return torch.zeros(shape, dtype=torch.float32)
            return torch.randn(shape, generator=rng, dtype=torch.float32) * 0.02

        self.fc1_weight_torch = init((ffn_dim, d_model))
        self.fc1_bias_torch = torch.zeros((ffn_dim,), dtype=torch.float32)
        self.fc2_weight_torch = init((d_model, ffn_dim))
        self.fc2_bias_torch = torch.zeros((d_model,), dtype=torch.float32)
        for attr in ("fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"):
            setattr(self, attr, upload(getattr(self, f"{attr}_torch"), device=device, dtype=dtype))

    def load_hf_state_dict(self, state: Mapping[str, torch.Tensor], *, strict: bool = True) -> LoadResult:
        return load_tensors(
            self,
            state,
            (
                ("fc1.weight", "fc1_weight"),
                ("fc1.bias", "fc1_bias"),
                ("fc2.weight", "fc2_weight"),
                ("fc2.bias", "fc2_bias"),
            ),
            device=self.device,
            dtype=self.dtype,
            strict=strict,
            label="feed-forward",
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        hidden = linear(x, self.fc1_weight, self.fc1_bias, dtype=self.dtype, memory_config=self.memory_config)
        hidden = activation(hidden, self.activation_function)
        return linear(hidden, self.fc2_weight, self.fc2_bias, dtype=self.dtype, memory_config=self.memory_config)


__all__ = ["FeedForward", "LayerNorm"]
