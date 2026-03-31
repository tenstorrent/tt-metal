# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Optional

import torch

import ttnn

from .config import TILE_SIZE
from .embeddings import ValueEmbedding
from .ops import apply_dropout, linear, make_causal_mask, slice_to_length
from .state_io import to_float_tensor
from .transformer import LayerNorm


def hf_sinusoidal_position_encoding(length: int, d_model: int) -> torch.Tensor:
    """Match HF InformerSinusoidalPositionalEmbedding (sin block then cos block)."""
    position = torch.arange(length, dtype=torch.float32)[:, None]
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    sin = torch.sin(position * div_term)
    cos = torch.cos(position * div_term)
    sentinel = d_model // 2 if d_model % 2 == 0 else (d_model // 2) + 1
    pe = torch.zeros((length, d_model), dtype=torch.float32)
    pe[:, 0:sentinel] = sin
    pe[:, sentinel:] = cos[:, : d_model - sentinel]
    return pe.unsqueeze(0)


class HfPositionalEmbedding:
    def __init__(self, max_len: int, d_model: int, *, device, dtype: ttnn.DataType):
        self.weight_torch = hf_sinusoidal_position_encoding(max_len, d_model)
        self.weight = ttnn.from_torch(self.weight_torch, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    def set_weight(self, weight: torch.Tensor, *, device, dtype: ttnn.DataType) -> None:
        if weight.dim() == 2:
            weight = weight.unsqueeze(0)
        self.weight_torch = weight.detach().float()
        self.weight = ttnn.from_torch(self.weight_torch, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)

    def __call__(self, length: int, *, offset: int = 0) -> ttnn.Tensor:
        if offset == 0:
            return slice_to_length(self.weight, dim=1, length=length)
        if offset + length > self.weight.shape[1]:
            raise ValueError(f"Positional embedding slice [{offset}:{offset + length}] exceeds max length.")
        weight_rm = ttnn.to_layout(self.weight, ttnn.ROW_MAJOR_LAYOUT)
        slice_start = [0] * len(weight_rm.shape)
        slice_end = list(weight_rm.shape)
        slice_start[1] = offset
        slice_end[1] = offset + length
        sliced = ttnn.slice(weight_rm, slice_start, slice_end)
        return ttnn.to_layout(sliced, ttnn.TILE_LAYOUT)


class HfEmbedding:
    def __init__(
        self,
        feature_size: int,
        d_model: int,
        rng: torch.Generator,
        *,
        device,
        dtype: ttnn.DataType,
        weight_dtype: Optional[ttnn.DataType] = None,
        compute_dtype: Optional[ttnn.DataType] = None,
        max_len: int,
        dropout: float,
        memory_config: Optional[ttnn.MemoryConfig] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.value_embedding = ValueEmbedding(
            feature_size,
            d_model,
            rng,
            device=device,
            dtype=dtype,
            weight_dtype=weight_dtype,
            use_bias=False,
            memory_config=memory_config,
        )
        self.positional_embedding = HfPositionalEmbedding(max_len, d_model, device=device, dtype=dtype)
        self.norm = LayerNorm(d_model, rng, device=device, dtype=dtype, compute_dtype=compute_dtype)
        self.dropout = dropout

    def load_hf_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool = True) -> dict[str, list[str]]:
        used: set[str] = set()
        missing: list[str] = []

        value_state = {}
        if "value_embedding.value_projection.weight" in state:
            value_state["weight"] = state["value_embedding.value_projection.weight"]
            used.add("value_embedding.value_projection.weight")
        value_result = self.value_embedding.load_hf_state_dict(value_state, strict=strict)
        missing.extend([f"value_embedding.value_projection.{k}" for k in value_result["missing_keys"]])

        pos_weight = state.get("embed_positions.weight")
        if pos_weight is None:
            missing.append("embed_positions.weight")
        else:
            used.add("embed_positions.weight")
            self.positional_embedding.set_weight(pos_weight, device=self.device, dtype=self.dtype)

        norm_state = {}
        if "layernorm_embedding.weight" in state:
            norm_state["weight"] = state["layernorm_embedding.weight"]
            used.add("layernorm_embedding.weight")
        if "layernorm_embedding.bias" in state:
            norm_state["bias"] = state["layernorm_embedding.bias"]
            used.add("layernorm_embedding.bias")
        norm_result = self.norm.load_hf_state_dict(norm_state, strict=strict)
        missing.extend([f"layernorm_embedding.{k}" for k in norm_result["missing_keys"]])

        unexpected = sorted(k for k in state if k not in used)
        if strict and missing:
            raise ValueError(f"Missing HF embedding weights: {missing}")
        return {"missing_keys": missing, "unexpected_keys": unexpected}

    def __call__(self, inputs: ttnn.Tensor, *, position_offset: int = 0) -> ttnn.Tensor:
        pos = self.positional_embedding(inputs.shape[1], offset=position_offset)
        x = self.value_embedding(inputs) + pos
        x = self.norm(x)
        return apply_dropout(x, self.dropout)


class ParameterProjection:
    def __init__(
        self,
        d_model: int,
        out_dim: int,
        num_params: int,
        rng: torch.Generator,
        *,
        device,
        dtype: ttnn.DataType,
        compute_dtype: Optional[ttnn.DataType] = None,
        weight_dtype: Optional[ttnn.DataType] = None,
        memory_config: Optional[ttnn.MemoryConfig] = None,
    ):
        self.output_dtype = compute_dtype or dtype
        self.weight_dtype = weight_dtype or dtype
        self.dtype = dtype
        self.device = device
        self.memory_config = memory_config
        self.num_params = num_params
        self.out_dim = out_dim
        self.weights_torch = [
            torch.randn((out_dim, d_model), generator=rng, dtype=torch.float32) * 0.02 for _ in range(num_params)
        ]
        self.biases_torch = [torch.zeros((out_dim,), dtype=torch.float32) for _ in range(num_params)]
        self.weights = [
            ttnn.from_torch(w, device=device, dtype=self.weight_dtype, layout=ttnn.TILE_LAYOUT)
            for w in self.weights_torch
        ]
        self.biases = [
            ttnn.from_torch(b, device=device, dtype=self.weight_dtype, layout=ttnn.TILE_LAYOUT)
            for b in self.biases_torch
        ]

    def set_weights(self, weights: list[torch.Tensor], biases: list[torch.Tensor]) -> None:
        if len(weights) != len(biases) or len(weights) != self.num_params:
            raise ValueError(
                f"ParameterProjection expected {self.num_params} weights/biases, got {len(weights)} / {len(biases)}."
            )
        self.weights_torch = [w.detach().float() for w in weights]
        self.biases_torch = [b.detach().float() for b in biases]
        self.weights = [
            ttnn.from_torch(w, device=self.device, dtype=self.weight_dtype, layout=ttnn.TILE_LAYOUT)
            for w in self.weights_torch
        ]
        self.biases = [
            ttnn.from_torch(b, device=self.device, dtype=self.weight_dtype, layout=ttnn.TILE_LAYOUT)
            for b in self.biases_torch
        ]

    def load_hf_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool = True) -> dict[str, list[str]]:
        used: set[str] = set()
        missing: list[str] = []
        weights: list[torch.Tensor] = []
        biases: list[torch.Tensor] = []
        for idx in range(self.num_params):
            w_key = f"proj.{idx}.weight"
            b_key = f"proj.{idx}.bias"
            weight = state.get(w_key)
            bias = state.get(b_key)
            if weight is None or bias is None:
                if weight is None:
                    missing.append(w_key)
                if bias is None:
                    missing.append(b_key)
                continue
            used.add(w_key)
            used.add(b_key)
            weights.append(to_float_tensor(weight))
            biases.append(to_float_tensor(bias))
        if len(weights) == self.num_params:
            self.set_weights(weights, biases)
        unexpected = sorted(k for k in state if k not in used)
        if strict and missing:
            raise ValueError(f"Missing parameter projection weights: {missing}")
        return {"missing_keys": missing, "unexpected_keys": unexpected}

    def __call__(self, x: ttnn.Tensor) -> list[ttnn.Tensor]:
        outputs = []
        for weight, bias in zip(self.weights, self.biases):
            outputs.append(linear(x, weight, bias, dtype=self.output_dtype, memory_config=self.memory_config))
        return outputs


def torch_squareplus(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + torch.sqrt(x * x + 4.0))


def torch_student_t_domain_map(
    df: torch.Tensor,
    loc: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = torch_squareplus(scale).clamp_min(torch.finfo(scale.dtype).eps)
    df = 2.0 + torch_squareplus(df)
    return df.squeeze(-1), loc.squeeze(-1), scale.squeeze(-1)


def torch_normal_domain_map(
    loc: torch.Tensor,
    scale: torch.Tensor,
    *,
    minimum_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch_squareplus(scale).clamp_min(max(minimum_scale, torch.finfo(scale.dtype).eps))
    return loc.squeeze(-1), scale.squeeze(-1)


def torch_negative_binomial_domain_map(
    mu: torch.Tensor,
    alpha: torch.Tensor,
    *,
    minimum_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    eps = max(minimum_scale, torch.finfo(mu.dtype).eps)
    mu = torch_squareplus(mu).clamp_min(eps)
    alpha = torch_squareplus(alpha).clamp_min(eps)
    total_count = 1.0 / alpha
    logits = -torch.log(mu * alpha)
    return total_count.squeeze(-1), logits.squeeze(-1)


def torch_mean_scaler(
    data: torch.Tensor,
    observed_indicator: torch.Tensor,
    *,
    minimum_scale: float,
    default_scale: Optional[float],
    dim: int = 1,
    keepdim: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ts_sum = (data * observed_indicator).abs().sum(dim, keepdim=True)
    num_observed = observed_indicator.sum(dim, keepdim=True)
    scale = ts_sum / torch.clamp(num_observed, min=1)
    if default_scale is None:
        batch_sum = ts_sum.sum(dim=0)
        batch_observations = torch.clamp(num_observed.sum(0), min=1)
        default_scale = torch.squeeze(batch_sum / batch_observations)
    else:
        default_scale = default_scale * torch.ones_like(scale)
    scale = torch.where(num_observed > 0, scale, default_scale)
    scale = torch.clamp(scale, min=minimum_scale)
    scaled_data = data / scale
    if not keepdim:
        scale = scale.squeeze(dim=dim)
    return scaled_data, torch.zeros_like(scale), scale


__all__ = [
    "HfEmbedding",
    "HfPositionalEmbedding",
    "ParameterProjection",
    "TILE_SIZE",
    "make_causal_mask",
    "torch_mean_scaler",
    "torch_negative_binomial_domain_map",
    "torch_normal_domain_map",
    "torch_student_t_domain_map",
]
