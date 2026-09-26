# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

import torch

import ttnn

from .ops import activation, linear
from .weights import LoadResult, merge_results, substate, to_float_tensor, upload


class Linear:
    """A weight and optional bias, with HuggingFace-compatible loading.

    Every projection in the model is one of these. Holding them as named attributes rather than
    assigning ``f"{name}_weight"`` through ``setattr`` keeps the modules readable and keeps
    static analysers from reading computed attribute names as dynamic code generation.
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        *,
        device,
        dtype: ttnn.DataType,
        use_bias: bool = True,
        memory_config: Optional[ttnn.MemoryConfig] = None,
        rng: Optional[torch.Generator] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.memory_config = memory_config

        if rng is None:
            self.weight_torch = torch.zeros((out_features, in_features), dtype=torch.float32)
        else:
            self.weight_torch = torch.randn((out_features, in_features), generator=rng, dtype=torch.float32) * 0.02
        self.weight = upload(self.weight_torch, device=device, dtype=dtype)

        self.bias_torch = torch.zeros((out_features,), dtype=torch.float32) if use_bias else None
        self.bias = upload(self.bias_torch, device=device, dtype=dtype) if use_bias else None

    def load_hf_state_dict(self, state: Mapping[str, torch.Tensor], *, strict: bool = True) -> LoadResult:
        missing: list[str] = []
        used: set[str] = set()

        weight = state.get("weight")
        if weight is None:
            missing.append("weight")
        else:
            used.add("weight")
            self.weight_torch = to_float_tensor(weight)
            self.weight = upload(self.weight_torch, device=self.device, dtype=self.dtype)

        if self.bias is not None:
            bias = state.get("bias")
            if bias is None:
                missing.append("bias")
            else:
                used.add("bias")
                self.bias_torch = to_float_tensor(bias)
                self.bias = upload(self.bias_torch, device=self.device, dtype=self.dtype)

        if strict and missing:
            raise ValueError(f"Missing linear weights: {missing}")
        return {"missing_keys": missing, "unexpected_keys": sorted(k for k in state if k not in used)}

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return linear(x, self.weight, self.bias, dtype=self.dtype, memory_config=self.memory_config)


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
        # ttnn.layer_norm expects the affine terms as row vectors.
        self.weight = upload(self.weight_torch.reshape(1, -1), device=device, dtype=dtype)
        self.bias = upload(self.bias_torch.reshape(1, -1), device=device, dtype=dtype)

    def load_hf_state_dict(self, state: Mapping[str, torch.Tensor], *, strict: bool = True) -> LoadResult:
        missing: list[str] = []
        used: set[str] = set()

        weight = state.get("weight")
        if weight is None:
            missing.append("weight")
        else:
            used.add("weight")
            self.weight_torch = to_float_tensor(weight)
            self.weight = upload(self.weight_torch.reshape(1, -1), device=self.device, dtype=self.dtype)

        bias = state.get("bias")
        if bias is None:
            missing.append("bias")
        else:
            used.add("bias")
            self.bias_torch = to_float_tensor(bias)
            self.bias = upload(self.bias_torch.reshape(1, -1), device=self.device, dtype=self.dtype)

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
        self.activation_function = activation_function
        self.fc1 = Linear(ffn_dim, d_model, device=device, dtype=dtype, memory_config=memory_config, rng=rng)
        self.fc2 = Linear(d_model, ffn_dim, device=device, dtype=dtype, memory_config=memory_config, rng=rng)

    def load_hf_state_dict(self, state: Mapping[str, torch.Tensor], *, strict: bool = True) -> LoadResult:
        return merge_results(
            [
                ("fc1", self.fc1.load_hf_state_dict(substate(state, "fc1"), strict=strict)),
                ("fc2", self.fc2.load_hf_state_dict(substate(state, "fc2"), strict=strict)),
            ],
            state=state,
            claimed=("fc1", "fc2"),
        )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.fc2(activation(self.fc1(x), self.activation_function))


__all__ = ["FeedForward", "LayerNorm", "Linear"]
