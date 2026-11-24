# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The FFN sublayer of a GLM-5.3-Flash decoder block, built from HF-named weights.

* :class:`GLMDenseMLP` -- ``Glm5NextTextMLP`` (layers 0-2, I = 12288; also the MoE's shared expert):
  ``down(silu(min(gate x, limit)) * clamp(up x, -limit, limit))``.
* :func:`build_moe` -- ``Glm5NextTextMoE`` on the sparse_matmul :class:`~models.experimental.glm53_flash.tt.moe.MoE`.

Both take ``x [1, 1, T, D]`` bf16 TILE and return the same shape.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from models.experimental.glm53_flash.tt.ds_v4.common import DeepSeekV4Module
from models.experimental.glm53_flash.tt.ds_v4.layers import Linear
from models.experimental.glm53_flash.tt.ds_v4.streams import _strip_prefix
from models.experimental.glm53_flash.tt.ds_v4.weight_cache import (
    WeightCache,
    _as_cache,
    _load_weight,
    _materialize,
    _memo,
)
from models.experimental.glm53_flash.tt.mlp import SwiGLUMLP
from models.experimental.glm53_flash.tt.moe import MoE, Router


class GLMDenseMLP(DeepSeekV4Module):
    """``weights``: ``gate_proj.weight`` / ``up_proj.weight`` ``[I, D]`` and ``down_proj.weight`` ``[D, I]``."""

    def __init__(
        self,
        config,
        weights: dict,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
    ):
        cache = _as_cache(cache)
        self.swiglu_limit = config.swiglu_limit
        self.gate_proj = Linear(weights["gate_proj.weight"], device, cache.file("gate_proj"), dtype=weight_dtype)
        self.up_proj = Linear(weights["up_proj.weight"], device, cache.file("up_proj"), dtype=weight_dtype)
        self.down_proj = Linear(weights["down_proj.weight"], device, cache.file("down_proj"), dtype=weight_dtype)

    def forward(self, x: ttnn.Tensor, decode: bool = False) -> ttnn.Tensor:
        gate, up = self.gate_proj(x), self.up_proj(x)
        gate = ttnn.clamp(gate, max=self.swiglu_limit)
        up = ttnn.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        gated = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        for tensor in (gate, up):
            ttnn.deallocate(tensor)
        out = self.down_proj(gated)
        ttnn.deallocate(gated)
        return out


class GLMMoE(DeepSeekV4Module):
    """``Glm5NextTextMoE``: the noaux_tc sigmoid router, the routed experts and the shared expert.

    ``weights`` (HF module names): ``gate.weight`` ``[E, D]``, ``gate.e_score_correction_bias`` ``[E]``,
    ``experts.gate_up_proj`` ``[E, 2I, D]`` (rows ``[gate | up]``), ``experts.down_proj`` ``[E, D, I]`` and
    ``shared_experts.{gate,up,down}_proj.weight``.
    """

    def __init__(
        self,
        config,
        weights: dict,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        expert_dtype: ttnn.DataType = ttnn.bfloat4_b,
    ):
        cache = _as_cache(cache)
        num_experts = config.n_routed_experts
        inter = config.moe_intermediate_size

        def load(name, thunk, dtype):
            file = cache.file(name)
            return _load_weight(_materialize(thunk, file, dtype), device, cache_file_name=file, dtype=dtype)

        gate_w = _memo(weights["gate.weight"])
        router = Router(
            device,
            load("gate", lambda: gate_w().detach().float().t().contiguous(), ttnn.bfloat16),
            _memo(weights["gate.e_score_correction_bias"])().detach().float(),
            num_experts=num_experts,
            top_k=config.num_experts_per_tok,
            scaling=config.routed_scaling_factor,
        )
        shared_w = _strip_prefix(weights, "shared_experts")
        shared_load = lambda name: load(
            f"shared_experts.{name}",
            lambda: _memo(shared_w[f"{name}.weight"])().detach().float().t().contiguous(),
            weight_dtype,
        )
        shared = SwiGLUMLP(
            device,
            shared_load("gate_proj"),
            shared_load("up_proj"),
            shared_load("down_proj"),
            limit=config.swiglu_limit,
        )
        gate_up = _memo(weights["experts.gate_up_proj"])
        down = _memo(weights["experts.down_proj"])
        self.moe = MoE(
            device,
            router,
            gate_up=load(
                "experts.gate_up_proj",
                lambda: gate_up().detach().float().transpose(1, 2).unsqueeze(0).contiguous(),
                expert_dtype,
            ),
            down=load(
                "experts.down_proj",
                lambda: down().detach().float().transpose(1, 2).unsqueeze(0).contiguous(),
                expert_dtype,
            ),
            shared=shared,
            intermediate=inter,
            hidden=config.hidden_size,
            limit=config.swiglu_limit,
        )

    def forward(self, x: ttnn.Tensor, decode: bool = False) -> ttnn.Tensor:
        return self.moe(x, decode=decode)


def build_ffn(
    config, layer_idx: int, weights: dict, device: ttnn.MeshDevice, cache: Optional[WeightCache] = None, **kwargs
):
    """The layer's FFN by ``config.mlp_layer_types[layer_idx]`` (``"sparse"`` -> MoE, else dense)."""
    if config.mlp_layer_types[layer_idx] == "sparse":
        return GLMMoE(config, weights, device, cache=cache, **kwargs)
    kwargs.pop("expert_dtype", None)
    return GLMDenseMLP(config, weights, device, cache=cache, **kwargs)
