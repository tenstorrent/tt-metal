# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import ttnn
from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.cpu_reference import rms_norm, swiglu_expert, v4_router
from models.demos.deepseek_v4_flash.real_attention_projection_smoke import (
    DEFAULT_ATTENTION_PROJECTION_MAX_BYTES,
    DEFAULT_ATTENTION_PROJECTION_MAX_TENSORS,
    _accuracy_summary,
    _metadata_summary,
    _tensor_summary,
    deterministic_attention_activation,
    layer_attention_projection_keys,
)
from models.demos.deepseek_v4_flash.real_checkpoint_loader import (
    RealCheckpointTensorIndex,
    TensorMetadata,
    layer_expert_mlp_keys,
    layer_shared_expert_mlp_keys,
)
from models.demos.deepseek_v4_flash.real_expert_smoke import decode_real_expert_weights
from models.demos.deepseek_v4_flash.real_kv_projection_smoke import (
    DEFAULT_KV_PROJECTION_MAX_BYTES,
    DEFAULT_KV_PROJECTION_MAX_TENSORS,
    KvProjectionWeights,
    decode_real_kv_projection_weights,
    layer_kv_projection_keys,
)
from models.demos.deepseek_v4_flash.real_prefill_attention_runtime_smoke import (
    _layer_attention_output_projection_keys,
    decode_real_prefill_attention_projection_weights,
)
from models.demos.deepseek_v4_flash.real_prefill_cache_prep_smoke import (
    apply_deepseek_v4_rotary,
    precompute_deepseek_v4_rope_frequencies,
)
from models.demos.deepseek_v4_flash.real_shared_expert_smoke import (
    DEFAULT_SHARED_EXPERT_MAX_BYTES,
    DEFAULT_SHARED_EXPERT_MAX_TENSORS,
    SHARED_EXPERT_TTNN_TILE_MULTIPLE,
    decode_real_shared_expert_weights,
)
from models.demos.deepseek_v4_flash.ttnn_attention_projection import (
    AttentionProjectionWeights,
    TtAttentionProjection,
    grouped_output_projection_a,
)
from models.demos.deepseek_v4_flash.ttnn_routed_expert import TtRoutedExpertMLP
from models.demos.deepseek_v4_flash.ttnn_shared_expert import TtSharedExpertMLP

REAL_TRACEABLE_DECODE_SMOKE_SCHEMA_VERSION = 1
DEFAULT_TRACEABLE_DECODE_LAYER = 3
DEFAULT_TRACEABLE_DECODE_SEQ_LEN = 32
DEFAULT_TRACEABLE_DECODE_CACHE_LEN = 64
DEFAULT_TRACEABLE_DECODE_STEPS = 1
DEFAULT_TRACEABLE_DECODE_ROUTED_TOPK_PREFIX: int | None = None
DEFAULT_TRACEABLE_DECODE_MAX_ROUTED_TOPK = 8
DEFAULT_ATTENTION_OUTPUT_PROJECTION_MAX_TENSORS = 4
DEFAULT_ATTENTION_OUTPUT_PROJECTION_MAX_BYTES = 96 * 1024 * 1024
DEFAULT_TRACEABLE_DECODE_MAX_TENSORS = (
    DEFAULT_ATTENTION_PROJECTION_MAX_TENSORS
    + DEFAULT_ATTENTION_OUTPUT_PROJECTION_MAX_TENSORS
    + DEFAULT_KV_PROJECTION_MAX_TENSORS
    + DEFAULT_SHARED_EXPERT_MAX_TENSORS
    + 3
    + 6 * DEFAULT_TRACEABLE_DECODE_MAX_ROUTED_TOPK
    + 1
)
DEFAULT_TRACEABLE_DECODE_MAX_BYTES = (
    DEFAULT_ATTENTION_PROJECTION_MAX_BYTES
    + DEFAULT_ATTENTION_OUTPUT_PROJECTION_MAX_BYTES
    + DEFAULT_KV_PROJECTION_MAX_BYTES
    + DEFAULT_SHARED_EXPERT_MAX_BYTES
    + 64 * 1024 * 1024 * DEFAULT_TRACEABLE_DECODE_MAX_ROUTED_TOPK
    + 4096
)
DEFAULT_TRACEABLE_DECODE_TRACE_REGION_SIZE = 64 * 1024 * 1024
ATTENTION_OUTPUT_PROJECTION_ATOL = 5e-1
ATTENTION_OUTPUT_PROJECTION_RTOL = 8e-2
TRACEABLE_DECODE_ATTENTION_QK_SOFTMAX_MODE = "traceable_fixed_cache_window_qk_softmax"
TRACEABLE_DECODE_ATTENTION_LEGACY_BLEND_MODE = "traceable_fixed_cache_window_q_plus_kv_blend"
TRACEABLE_DECODE_ATTENTION_MODES = (
    TRACEABLE_DECODE_ATTENTION_QK_SOFTMAX_MODE,
    TRACEABLE_DECODE_ATTENTION_LEGACY_BLEND_MODE,
)
DEFAULT_TRACEABLE_DECODE_ATTENTION_MODE = TRACEABLE_DECODE_ATTENTION_QK_SOFTMAX_MODE
TRACEABLE_DECODE_CACHE_UPDATE_HOST_SCALAR = "update_cache"
TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR = "paged_update_cache"
TRACEABLE_DECODE_CACHE_UPDATE_APIS = (
    TRACEABLE_DECODE_CACHE_UPDATE_HOST_SCALAR,
    TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR,
)
DEFAULT_TRACEABLE_DECODE_CACHE_UPDATE_API = TRACEABLE_DECODE_CACHE_UPDATE_HOST_SCALAR
TRACEABLE_DECODE_ROPE_POSITION_STATIC = "static_tables"
TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR = "embedding_position_tensor"
TRACEABLE_DECODE_ROPE_POSITION_APIS = (
    TRACEABLE_DECODE_ROPE_POSITION_STATIC,
    TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR,
)
DEFAULT_TRACEABLE_DECODE_ROPE_POSITION_API = TRACEABLE_DECODE_ROPE_POSITION_STATIC


@dataclass(frozen=True)
class TraceableDecodeWeights:
    attention: AttentionProjectionWeights
    kv: KvProjectionWeights
    attn_norm: torch.Tensor
    ffn_norm: torch.Tensor
    router_gate: torch.Tensor
    router_bias: torch.Tensor | None
    router_tid2eid: torch.Tensor | None
    shared_expert: dict[str, torch.Tensor]
    routed_experts: dict[int, dict[str, torch.Tensor]]


@dataclass(frozen=True)
class TraceableDecodeRoutePlan:
    decode_token_index: int
    topk_prefix: int
    full_topk: int
    router_weights: torch.Tensor
    router_indices: torch.Tensor
    selected_weights: torch.Tensor
    selected_indices: torch.Tensor
    input_ids: torch.Tensor | None
    per_expert_route_weight: dict[int, torch.Tensor]

    @property
    def selected_expert_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.selected_indices.reshape(-1).tolist())


@dataclass(frozen=True)
class GuardedSymbol:
    module_name: str
    attr_path: str
    label: str


class TraceableDecodeHostFallbackError(RuntimeError):
    """Raised when a protected traceable decode forward crosses a host boundary."""


class TraceableDecodeHostGuard(AbstractContextManager["TraceableDecodeHostGuard"]):
    """Patch known host readback and DeepSeek fallback helpers during protected forwards."""

    def __init__(self, symbols: Sequence[GuardedSymbol] = ()):
        self._symbols = tuple(symbols) if symbols else default_guarded_symbols()
        self._patches: list[tuple[object, str, object]] = []
        self.guarded_labels: list[str] = []

    def __enter__(self) -> "TraceableDecodeHostGuard":
        for symbol in self._symbols:
            parent, attr = _resolve_patch_target(symbol)
            original = getattr(parent, attr)
            setattr(parent, attr, _blocked_host_boundary(symbol.label))
            self._patches.append((parent, attr, original))
            self.guarded_labels.append(symbol.label)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for parent, attr, original in reversed(self._patches):
            setattr(parent, attr, original)
        self._patches.clear()


class TtTraceableDecodeSubpath:
    """TTNN-only decode subpath suitable for trace capture.

    This is intentionally not a full decoder layer. The protected forward covers
    the device-resident query projection, compressed K/V projection and cache
    append, a fixed cache-window QK softmax stepping stone with traceable Q/K
    RoPE, grouped attention output projection, post-attention residual, and
    preselected routed/shared FFN stepping stone:
    ``hidden -> attn_norm -> wq_a -> q_norm -> wq_b``,
    ``attn_norm -> wkv -> kv_norm -> update_cache -> fixed cache-window read``,
    ``Q/K split -> Q/K RoPE -> fixed-window softmax/context -> grouped wo_a -> wo_b -> residual``, and
    ``post_attention_residual -> ffn_norm -> routed/shared experts -> residual``.
    """

    def __init__(
        self,
        *,
        device,
        weights: TraceableDecodeWeights,
        route_plan: TraceableDecodeRoutePlan,
        config: DeepSeekV4FlashConfig,
        layer: int,
        cache_len: int,
        cache_update_index: int,
        cache_update_api: str = DEFAULT_TRACEABLE_DECODE_CACHE_UPDATE_API,
        cache_update_idxs_tensor=None,
        rope_position_api: str = DEFAULT_TRACEABLE_DECODE_ROPE_POSITION_API,
        rope_position_idxs_tensor=None,
        attention_cache_window_index: int | None = None,
        rope_position_index: int | None = None,
        initial_kv_cache: torch.Tensor | None = None,
        kv_cache=None,
        attention_mode: str = DEFAULT_TRACEABLE_DECODE_ATTENTION_MODE,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        self.device = device
        self.config = config
        self.dtype = dtype
        self.memory_config = memory_config
        self.layer = int(layer)
        self.cache_len = int(cache_len)
        self.cache_update_index = int(cache_update_index)
        self.cache_update_api = _validate_cache_update_api(cache_update_api)
        self.cache_update_idxs_tensor = cache_update_idxs_tensor
        self.rope_position_api = _validate_rope_position_api(rope_position_api)
        self.rope_position_idxs_tensor = rope_position_idxs_tensor
        self.attention_cache_window_index = (
            int(cache_update_index) if attention_cache_window_index is None else int(attention_cache_window_index)
        )
        self.rope_position_index = (
            int(self.attention_cache_window_index) if rope_position_index is None else int(rope_position_index)
        )
        if self.cache_update_api == TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR and cache_update_idxs_tensor is None:
            raise ValueError("cache_update_idxs_tensor is required for paged_update_cache trace replay")
        if self.rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR and rope_position_idxs_tensor is None:
            raise ValueError("rope_position_idxs_tensor is required for dynamic RoPE position trace replay")
        self.attention_mode = _validate_attention_mode(attention_mode)
        self.route_plan = route_plan
        self.static_token_rows = _route_plan_static_token_rows(route_plan)
        self.attention = TtAttentionProjection(
            device=device,
            weights=weights.attention,
            hidden_size=int(config.hidden_size),
            q_lora_rank=int(config.q_lora_rank),
            num_heads=int(config.num_attention_heads),
            head_dim=int(config.head_dim),
            norm_eps=float(config.rms_norm_eps),
            o_groups=int(config.o_groups),
            o_lora_rank=int(config.o_lora_rank),
            dtype=dtype,
            memory_config=memory_config,
        )
        self.shared_expert = TtSharedExpertMLP(
            device=device,
            w1=weights.shared_expert["w1"],
            w2=weights.shared_expert["w2"],
            w3=weights.shared_expert["w3"],
            dtype=dtype,
            memory_config=memory_config,
            swiglu_limit=float(config.swiglu_limit),
        )
        self.routed_experts: dict[int, TtRoutedExpertMLP] = {}
        for expert in route_plan.selected_expert_ids:
            if expert not in weights.routed_experts:
                raise KeyError(f"Missing routed expert weights for selected expert {expert}")
            expert_weights = weights.routed_experts[expert]
            routed_module = TtRoutedExpertMLP(
                device=device,
                w1=expert_weights["w1"],
                w2=expert_weights["w2"],
                w3=expert_weights["w3"],
                dtype=dtype,
                memory_config=memory_config,
                swiglu_limit=float(config.swiglu_limit),
            )
            self.routed_experts[expert] = routed_module
        self.kv_output_dim = _kv_output_dim(config)
        self.attention_cache_repeat_factor = _attention_cache_repeat_factor(config)
        self.router_mode = _router_mode(config=config, weights=weights)
        self.router_gate = _to_tt_linear_weight(
            weights.router_gate,
            device=device,
            dtype=dtype,
            memory_config=memory_config,
        )
        self.router_bias = (
            _to_tt_router_bias(
                weights.router_bias,
                token_rows=self.static_token_rows,
                device=device,
                dtype=dtype,
                memory_config=memory_config,
            )
            if weights.router_bias is not None
            else None
        )
        self.static_router_indices = (
            _to_tt_static_router_indices(
                route_plan.router_indices,
                token_rows=self.static_token_rows,
                device=device,
                memory_config=memory_config,
            )
            if weights.router_tid2eid is not None
            else None
        )
        self.router_row_mask = _to_tt_router_decode_row_mask(
            token_rows=self.static_token_rows,
            decode_token_index=route_plan.decode_token_index,
            device=device,
            dtype=dtype,
            memory_config=memory_config,
        )
        self.wkv = _to_tt_linear_weight(
            weights.kv.wkv,
            device=device,
            dtype=dtype,
            memory_config=memory_config,
        )
        self.kv_norm = _to_tt_norm_weight(
            weights.kv.kv_norm,
            device=device,
            dtype=dtype,
            memory_config=memory_config,
        )
        if kv_cache is not None and initial_kv_cache is not None:
            raise ValueError("Pass either initial_kv_cache or kv_cache, not both")
        self.kv_cache = (
            kv_cache
            if kv_cache is not None
            else _to_tt_kv_cache(
                cache_len=self.cache_len,
                kv_output_dim=self.kv_output_dim,
                initial_cache=initial_kv_cache,
                device=device,
                dtype=dtype,
                memory_config=memory_config,
            )
        )
        self.attn_norm = _to_tt_norm_weight(
            weights.attn_norm,
            device=device,
            dtype=dtype,
            memory_config=memory_config,
        )
        self.ffn_norm = _to_tt_norm_weight(
            weights.ffn_norm,
            device=device,
            dtype=dtype,
            memory_config=memory_config,
        )
        self.q_head_norm = _to_tt_norm_weight(
            torch.ones(int(config.head_dim), dtype=torch.bfloat16),
            device=device,
            dtype=dtype,
            memory_config=memory_config,
        )
        if self.rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR:
            self.rope_cos_table, self.rope_sin_table, self.rope_trans_mat = _to_tt_rope_embedding_tables(
                config,
                layer=self.layer,
                max_position=self.cache_len,
                device=device,
                dtype=dtype,
                memory_config=memory_config,
            )
            self.rope_cos = None
            self.rope_sin = None
        else:
            self.rope_cos, self.rope_sin, self.rope_trans_mat = _to_tt_rope_tensors(
                config,
                layer=self.layer,
                start_pos=self.rope_position_index,
                seq_len=self.static_token_rows,
                device=device,
                dtype=dtype,
                memory_config=memory_config,
            )
            self.rope_cos_table = None
            self.rope_sin_table = None

    def __call__(self, hidden_states) -> dict[str, object]:
        _validate_ttnn_hidden_states(hidden_states, hidden_size=int(self.config.hidden_size))
        attn_norm_output = ttnn.rms_norm(
            hidden_states,
            weight=self.attn_norm,
            epsilon=float(self.config.rms_norm_eps),
            memory_config=self.memory_config,
        )
        q_rank_norm = self.attention.project_q_rank(attn_norm_output)
        q_output = self.attention.project_q_from_rank(q_rank_norm)
        kv_linear = ttnn.linear(attn_norm_output, self.wkv, memory_config=self.memory_config)
        kv_output = ttnn.rms_norm(
            kv_linear,
            weight=self.kv_norm,
            epsilon=float(self.config.rms_norm_eps),
            memory_config=self.memory_config,
        )
        if self.cache_update_api == TRACEABLE_DECODE_CACHE_UPDATE_HOST_SCALAR:
            kv_update = ttnn.to_memory_config(
                kv_output,
                _kv_update_memory_config(
                    device=self.device, token_rows=int(kv_output.shape[-2]), width=self.kv_output_dim
                ),
            )
            self.kv_cache = ttnn.update_cache(self.kv_cache, kv_update, self.cache_update_index)
        else:
            kv_update_decode_token = ttnn.slice(
                kv_output,
                (0, 0, 0, 0),
                (1, 1, 1, self.kv_output_dim),
                memory_config=self.memory_config,
            )
            kv_update = ttnn.to_memory_config(
                kv_update_decode_token,
                _single_core_height_sharded_memory_config(kv_update_decode_token, device=self.device),
            )
            self.kv_cache = ttnn.experimental.paged_update_cache(
                self.kv_cache,
                kv_update,
                update_idxs_tensor=self.cache_update_idxs_tensor,
            )
        attention_cache_window = ttnn.slice(
            self.kv_cache,
            (0, 0, self.attention_cache_window_index, 0),
            (1, 1, self.attention_cache_window_index + int(hidden_states.shape[-2]), self.kv_output_dim),
            memory_config=self.memory_config,
        )
        if self.rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR:
            rope_cos, rope_sin = _ttnn_gather_rope_tensors(
                self.rope_position_idxs_tensor,
                self.rope_cos_table,
                self.rope_sin_table,
                memory_config=self.memory_config,
            )
        else:
            rope_cos = self.rope_cos
            rope_sin = self.rope_sin
        attention_intermediates = _ttnn_fixed_window_attention(
            q_output=q_output,
            attention_cache_window=attention_cache_window,
            attention_mode=self.attention_mode,
            config=self.config,
            q_head_norm=self.q_head_norm,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            rope_trans_mat=self.rope_trans_mat,
            memory_config=self.memory_config,
        )
        attention_output = attention_intermediates["attention_output"]
        attention_projected = self.attention.project_output(attention_output)
        post_attention_residual = ttnn.add(hidden_states, attention_projected, memory_config=self.memory_config)
        ffn_norm_output = ttnn.rms_norm(
            post_attention_residual,
            weight=self.ffn_norm,
            epsilon=float(self.config.rms_norm_eps),
            memory_config=self.memory_config,
        )
        router_outputs = _ttnn_router_path(
            ffn_norm_output,
            router_gate=self.router_gate,
            router_bias=self.router_bias,
            static_router_indices=self.static_router_indices,
            config=self.config,
            memory_config=self.memory_config,
        )
        router_selected_route_weights = _ttnn_topk_prefix(
            router_outputs["router_route_weights"],
            topk_prefix=self.route_plan.topk_prefix,
            memory_config=self.memory_config,
        )
        router_selected_route_weights_masked = ttnn.mul(
            router_selected_route_weights,
            self.router_row_mask,
            memory_config=self.memory_config,
        )
        router_decode_topk_indices = _ttnn_decode_token_slice(
            router_outputs["router_topk_indices"],
            decode_token_index=self.route_plan.decode_token_index,
            memory_config=self.memory_config,
        )
        router_decode_route_weights = _ttnn_decode_token_slice(
            router_outputs["router_route_weights"],
            decode_token_index=self.route_plan.decode_token_index,
            memory_config=self.memory_config,
        )
        router_decode_selected_route_weights = _ttnn_decode_token_slice(
            router_selected_route_weights,
            decode_token_index=self.route_plan.decode_token_index,
            memory_config=self.memory_config,
        )
        router_outputs = {
            **router_outputs,
            "router_selected_route_weights": router_selected_route_weights,
            "router_selected_route_weights_masked": router_selected_route_weights_masked,
            "router_decode_topk_indices": router_decode_topk_indices,
            "router_decode_route_weights": router_decode_route_weights,
            "router_decode_selected_route_weights": router_decode_selected_route_weights,
        }
        shared_output = self.shared_expert(ffn_norm_output)
        routed_expert_outputs = {}
        for slot, (expert, routed_module) in enumerate(self.routed_experts.items()):
            route_weight = _ttnn_route_weight_for_topk_slot(
                router_outputs["router_route_weights"],
                slot=slot,
                route_mask=self.router_row_mask,
                intermediate_size=routed_module.intermediate_size,
                memory_config=self.memory_config,
            )
            routed_expert_outputs[f"routed_expert_{expert}_output"] = routed_module(
                ffn_norm_output,
                route_weight=route_weight,
            )
        routed_output = _sum_ttnn_tensors(
            routed_expert_outputs.values(),
            memory_config=self.memory_config,
        )
        combined_ffn_output = ttnn.add(shared_output, routed_output, memory_config=self.memory_config)
        residual_output = ttnn.add(post_attention_residual, combined_ffn_output, memory_config=self.memory_config)
        return {
            "attn_norm_output": attn_norm_output,
            "q_rank_norm": q_rank_norm,
            "q_output": q_output,
            "kv_linear": kv_linear,
            "kv_output": kv_output,
            "kv_cache": self.kv_cache,
            "attention_cache_window": attention_cache_window,
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
            **attention_intermediates,
            "attention_projected": attention_projected,
            "post_attention_residual": post_attention_residual,
            "ffn_norm_output": ffn_norm_output,
            **router_outputs,
            "shared_output": shared_output,
            **routed_expert_outputs,
            "routed_output": routed_output,
            "combined_ffn_output": combined_ffn_output,
            "residual_output": residual_output,
        }


def run_traceable_decode_subpath_smoke(
    snapshot_dir: str | Path,
    *,
    layer: int = DEFAULT_TRACEABLE_DECODE_LAYER,
    seq_len: int = DEFAULT_TRACEABLE_DECODE_SEQ_LEN,
    routed_topk_prefix: int | None = DEFAULT_TRACEABLE_DECODE_ROUTED_TOPK_PREFIX,
    max_tensors: int = DEFAULT_TRACEABLE_DECODE_MAX_TENSORS,
    max_bytes: int = DEFAULT_TRACEABLE_DECODE_MAX_BYTES,
    cpu_only: bool = False,
    device_id: int = 0,
    trace_region_size: int = DEFAULT_TRACEABLE_DECODE_TRACE_REGION_SIZE,
    cache_len: int = DEFAULT_TRACEABLE_DECODE_CACHE_LEN,
    cache_update_index: int | None = None,
    decode_steps: int = DEFAULT_TRACEABLE_DECODE_STEPS,
    attention_mode: str = DEFAULT_TRACEABLE_DECODE_ATTENTION_MODE,
    cache_update_api: str = DEFAULT_TRACEABLE_DECODE_CACHE_UPDATE_API,
    rope_position_api: str = DEFAULT_TRACEABLE_DECODE_ROPE_POSITION_API,
    pcc: float = 0.99,
    rtol: float = 8e-2,
    atol: float = 8e-2,
) -> dict[str, Any]:
    """Load real weights and optionally trace/replay the protected TTNN decode subpath."""

    _validate_smoke_args(
        layer=layer,
        seq_len=seq_len,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
        trace_region_size=trace_region_size,
        cache_len=cache_len,
        cache_update_index=cache_update_index,
        decode_steps=decode_steps,
        routed_topk_prefix=routed_topk_prefix,
        pcc=pcc,
    )
    cache_update_index = _resolve_cache_update_index(
        seq_len=seq_len,
        cache_len=cache_len,
        cache_update_index=cache_update_index,
    )
    attention_mode = _validate_attention_mode(attention_mode)
    cache_update_api = _validate_cache_update_api(cache_update_api)
    rope_position_api = _validate_rope_position_api(rope_position_api)
    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    config = DeepSeekV4FlashConfig.from_model_path(snapshot_dir)
    tensors, metadata, keys = load_traceable_decode_subpath_slice(
        snapshot_dir,
        layer=layer,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    weights = decode_traceable_decode_subpath_weights(tensors, config=config, layer=layer)
    activation = deterministic_attention_activation(hidden_size=int(config.hidden_size), seq_len=seq_len)
    kv_cache_initial = deterministic_traceable_kv_cache_seed(
        cache_len=cache_len,
        kv_output_dim=_kv_output_dim(config),
        cache_update_index=cache_update_index,
        seq_len=seq_len,
        decode_steps=decode_steps,
    )
    activations = tuple(_decode_step_activation(activation, step) for step in range(int(decode_steps)))
    replay_activations = tuple(_replay_activation(step_activation) for step_activation in activations)
    cache_update_indices = tuple(int(cache_update_index) + step for step in range(int(decode_steps)))
    static_attention_index = int(cache_update_indices[0])
    attention_window_indices = (
        tuple(static_attention_index for _ in cache_update_indices)
        if cache_update_api == TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR
        else cache_update_indices
    )
    rope_position_indices = (
        cache_update_indices
        if rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR
        else attention_window_indices
    )

    route_plans: list[TraceableDecodeRoutePlan] = []
    preliminary_cache = kv_cache_initial
    for step, step_activation in enumerate(activations):
        preliminary_reference = build_torch_traceable_decode_subpath_reference(
            weights,
            config=config,
            layer=layer,
            activation=step_activation,
            kv_cache_initial=preliminary_cache,
            cache_len=cache_len,
            cache_update_index=cache_update_indices[step],
            attention_cache_window_index=attention_window_indices[step],
            rope_position_index=rope_position_indices[step],
            attention_mode=attention_mode,
            route_plan=None,
        )
        route_plans.append(
            build_traceable_decode_route_plan(
                weights,
                config=config,
                seq_len=seq_len,
                ffn_norm_output=preliminary_reference["ffn_norm_output"],
                routed_topk_prefix=routed_topk_prefix,
            )
        )
        preliminary_cache = preliminary_reference["kv_cache"]
        if cache_update_api == TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR:
            route_plans.extend([route_plans[0] for _ in range(1, int(decode_steps))])
            break

    selected_expert_ids = _unique_ints(
        expert for route_plan in route_plans for expert in route_plan.selected_expert_ids
    )
    if selected_expert_ids:
        index = RealCheckpointTensorIndex.from_snapshot(snapshot_dir)
        routed_expert_keys = [
            key for expert in selected_expert_ids for key in layer_expert_mlp_keys(index, layer=layer, expert=expert)
        ]
        keys["routed_experts"] = routed_expert_keys
        tensors, metadata = _load_missing_tensors(
            index,
            tensors,
            metadata,
            routed_expert_keys,
            max_tensors=max_tensors,
            max_bytes=max_bytes,
        )
        weights = decode_traceable_decode_subpath_weights(
            tensors,
            config=config,
            layer=layer,
            selected_experts=selected_expert_ids,
        )
    references: list[dict[str, torch.Tensor]] = []
    replay_references: list[dict[str, torch.Tensor]] = []
    reference_cache = kv_cache_initial
    replay_reference_cache = kv_cache_initial
    for step, route_plan in enumerate(route_plans):
        reference = build_torch_traceable_decode_subpath_reference(
            weights,
            config=config,
            layer=layer,
            activation=activations[step],
            kv_cache_initial=reference_cache,
            cache_len=cache_len,
            cache_update_index=cache_update_indices[step],
            attention_cache_window_index=attention_window_indices[step],
            rope_position_index=rope_position_indices[step],
            attention_mode=attention_mode,
            route_plan=route_plan,
        )
        references.append(reference)
        reference_cache = reference["kv_cache"]
        replay_reference = build_torch_traceable_decode_subpath_reference(
            weights,
            config=config,
            layer=layer,
            activation=replay_activations[step],
            kv_cache_initial=replay_reference_cache,
            cache_len=cache_len,
            cache_update_index=cache_update_indices[step],
            attention_cache_window_index=attention_window_indices[step],
            rope_position_index=rope_position_indices[step],
            attention_mode=attention_mode,
            route_plan=route_plan,
        )
        replay_references.append(replay_reference)
        replay_reference_cache = replay_reference["kv_cache"]
    metadata_groups = _metadata_groups(metadata, keys)
    result = _base_result(
        snapshot_dir=snapshot_dir,
        layer=layer,
        seq_len=seq_len,
        cache_len=cache_len,
        cache_update_index=cache_update_index,
        cache_update_indices=cache_update_indices,
        decode_steps=decode_steps,
        config=config,
        metadata=metadata,
        metadata_groups=metadata_groups,
        weights=weights,
        activation=activations[0],
        kv_cache_initial=kv_cache_initial,
        replay_activation=replay_activations[0],
        activations=activations,
        replay_activations=replay_activations,
        route_plans=route_plans,
        reference=references[0],
        replay_reference=replay_references[0],
        references=references,
        replay_references=replay_references,
        attention_mode=attention_mode,
        cache_update_api=cache_update_api,
        rope_position_api=rope_position_api,
        attention_window_indices=attention_window_indices,
        rope_position_indices=rope_position_indices,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
        trace_region_size=trace_region_size,
    )

    if cpu_only:
        result["mode"] = "cpu-reference"
        result["accuracy"] = {
            "cpu_reference": {
                "passed": True,
                "reason": "cpu-only requested; TTNN trace capture was not run",
            }
        }
        result["accuracy_by_step"] = [
            {
                "step": step,
                "position": cache_update_indices[step],
                "accuracy": {
                    "cpu_reference": {
                        "passed": True,
                        "reason": "cpu-only requested; TTNN trace capture was not run",
                    }
                },
            }
            for step in range(int(decode_steps))
        ]
        for step_detail in result["decode_steps_detail"]:
            step_detail["accuracy"] = result["accuracy_by_step"][step_detail["step"]]["accuracy"]
        result["passed"] = True
        return result

    if seq_len % SHARED_EXPERT_TTNN_TILE_MULTIPLE != 0:
        raise ValueError(f"TTNN traceable decode seq_len must be a multiple of 32, got {seq_len}")

    ttnn_outputs_by_step, trace_info = _run_ttnn_traceable_decode_subpath(
        weights,
        route_plans=route_plans,
        config=config,
        activations=activations,
        replay_activations=replay_activations,
        kv_cache_initial=kv_cache_initial,
        device_id=device_id,
        trace_region_size=trace_region_size,
        layer=layer,
        cache_len=cache_len,
        cache_update_indices=cache_update_indices,
        attention_mode=attention_mode,
        cache_update_api=cache_update_api,
        rope_position_api=rope_position_api,
        attention_window_index=static_attention_index,
        rope_position_indices=rope_position_indices,
    )
    result["mode"] = "ttnn-trace"
    result["device_id"] = int(device_id)
    result["trace_capture"].update(trace_info)
    result["one_trace_capture_replayed_across_positions"] = bool(trace_info["single_capture_replayed_across_positions"])
    result["cache_update"]["single_capture_replay_across_positions"] = bool(
        trace_info["single_capture_replayed_across_positions"]
    )
    result["multi_position_replay"]["single_capture_replayed_across_positions"] = bool(
        trace_info["single_capture_replayed_across_positions"]
    )
    result["multi_position_replay"]["recaptured_per_position"] = bool(trace_info["recaptured_per_position"])
    result["multi_position_replay"]["cache_update_index_dynamic"] = bool(trace_info["cache_update_index_dynamic"])
    result["multi_position_replay"]["rope_position_dynamic"] = bool(trace_info["rope_position_dynamic"])
    result["rope_position_dynamic"] = bool(trace_info["rope_position_dynamic"])
    result["rope_position_status"] = trace_info["rope_position_status"]
    result["rope_position_kind"] = trace_info["rope_position_kind"]
    for step_detail in result["decode_steps_detail"]:
        step_detail["cache_update_index_kind"] = trace_info["cache_update_index_kind"]
        step_detail["rope_position_kind"] = trace_info["rope_position_kind"]
        step_detail["single_capture_replayed_across_this_position"] = bool(
            trace_info["single_capture_replayed_across_positions"]
        )
    result["trace_capture_attempted"] = bool(result["trace_capture"]["attempted"])
    result["trace_capture_passed"] = bool(result["trace_capture"]["capture_passed"])
    result["trace_execute_replay_passed"] = bool(result["trace_capture"]["execute_replay_passed"])
    result["guard_status"] = _guard_status(result["trace_capture"])
    result["ttnn_ops"] = _traceable_decode_ttnn_ops(
        attention_mode,
        cache_update_api=cache_update_api,
        config=config,
        weights=weights,
        route_plan=route_plans[0],
        rope_position_api=rope_position_api,
    )
    result["trace_capture"]["traced_operations"] = list(result["ttnn_ops"])
    result["ttnn"] = {name: _tensor_summary(value) for name, value in ttnn_outputs_by_step[0].items()}
    result["ttnn_by_step"] = [
        {
            "step": step,
            "position": cache_update_indices[step],
            "tensors": {name: _tensor_summary(value) for name, value in step_outputs.items()},
        }
        for step, step_outputs in enumerate(ttnn_outputs_by_step)
    ]
    result["accuracy_by_step"] = []
    for step, (step_outputs, step_reference) in enumerate(zip(ttnn_outputs_by_step, replay_references)):
        step_accuracy = {}
        for name, expected in _traceable_accuracy_items(step_reference):
            accuracy = _traceable_accuracy_summary(
                name,
                expected,
                step_outputs[name],
                pcc_threshold=pcc,
                rtol=rtol,
                atol=atol,
            )
            accuracy["required_for_pass"] = _traceable_accuracy_required_for_pass(name)
            step_accuracy[name] = accuracy
        result["accuracy_by_step"].append(
            {
                "step": step,
                "position": cache_update_indices[step],
                "accuracy": step_accuracy,
            }
        )
        result["decode_steps_detail"][step]["accuracy"] = step_accuracy
    result["accuracy"] = result["accuracy_by_step"][0]["accuracy"]
    if "router_decode_topk_indices" in result["accuracy"]:
        result["router_trace"]["device_topk_indices_match_torch_reference"] = bool(
            result["accuracy"]["router_decode_topk_indices"].get("allclose", False)
        )
        result["router_trace"]["topk_indices_accuracy_required_for_pass"] = bool(
            result["accuracy"]["router_decode_topk_indices"]["required_for_pass"]
        )
    if "router_decode_route_weights" in result["accuracy"]:
        result["router_trace"]["device_route_weights_match_torch_reference_allclose"] = bool(
            result["accuracy"]["router_decode_route_weights"].get("allclose", False)
        )
    result["passed"] = bool(
        result["trace_capture"]["attempted"]
        and result["trace_capture"]["capture_passed"]
        and result["trace_capture"]["execute_replay_passed"]
        and all(
            item["passed"]
            for step in result["accuracy_by_step"]
            for item in step["accuracy"].values()
            if bool(item.get("required_for_pass", True))
        )
    )
    return result


def load_traceable_decode_subpath_slice(
    snapshot_dir: str | Path,
    *,
    layer: int,
    routed_experts: Sequence[int] = (),
    max_tensors: int = DEFAULT_TRACEABLE_DECODE_MAX_TENSORS,
    max_bytes: int = DEFAULT_TRACEABLE_DECODE_MAX_BYTES,
) -> tuple[dict[str, torch.Tensor], list[TensorMetadata], dict[str, list[str]]]:
    index = RealCheckpointTensorIndex.from_snapshot(snapshot_dir)
    attention_keys = layer_attention_projection_keys(index, layer=layer)
    attention_output_keys = _layer_attention_output_projection_keys(index, layer=layer)
    kv_keys = [key for key in layer_kv_projection_keys(index, layer=layer) if key not in attention_keys]
    ffn_norm_keys = [f"layers.{layer}.ffn_norm.weight"]
    for key in ffn_norm_keys:
        index.location(key)
    router_keys = layer_traceable_decode_router_keys(index, layer=layer)
    shared_expert_keys = layer_shared_expert_mlp_keys(index, layer=layer)
    routed_expert_keys = [
        key for expert in routed_experts for key in layer_expert_mlp_keys(index, layer=layer, expert=int(expert))
    ]
    keys = {
        "attention_query": attention_keys,
        "attention_output": attention_output_keys,
        "kv_projection": kv_keys,
        "ffn_norm": ffn_norm_keys,
        "router_selector": router_keys,
        "shared_expert": shared_expert_keys,
        "routed_experts": routed_expert_keys,
    }
    tensors, metadata = index.load_tensors(
        _unique_keys(
            [
                *attention_keys,
                *attention_output_keys,
                *kv_keys,
                *ffn_norm_keys,
                *router_keys,
                *shared_expert_keys,
                *routed_expert_keys,
            ]
        ),
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    return tensors, metadata, keys


def layer_traceable_decode_router_keys(index: RealCheckpointTensorIndex, *, layer: int) -> list[str]:
    if layer < 0:
        raise ValueError(f"layer must be non-negative, got {layer}")
    prefix = f"layers.{layer}"
    keys = [f"{prefix}.ffn.gate.weight"]
    bias_key = f"{prefix}.ffn.gate.bias"
    tid2eid_key = f"{prefix}.ffn.gate.tid2eid"
    if index.has_tensor(bias_key):
        keys.append(bias_key)
    elif index.has_tensor(tid2eid_key):
        keys.append(tid2eid_key)
    else:
        raise KeyError(f"Layer {layer} has neither router bias nor tid2eid tensor in {index.snapshot_dir}")
    return keys


def decode_traceable_decode_subpath_weights(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    selected_experts: Sequence[int] = (),
) -> TraceableDecodeWeights:
    attention = decode_real_prefill_attention_projection_weights(tensors, config=config, layer=layer)
    kv = decode_real_kv_projection_weights(tensors, config=config, layer=layer)
    _validate_traceable_decode_router_tensors(tensors, config=config, layer=layer)
    ffn_norm_key = f"layers.{layer}.ffn_norm.weight"
    if ffn_norm_key not in tensors:
        raise KeyError(f"Missing required FFN norm tensor {ffn_norm_key!r}")
    if tuple(tensors[ffn_norm_key].shape) != (int(config.hidden_size),):
        raise ValueError(
            f"Expected {ffn_norm_key} shape {(int(config.hidden_size),)}, " f"got {tuple(tensors[ffn_norm_key].shape)}"
        )
    shared_expert = decode_real_shared_expert_weights(tensors, config=config, layer=layer)
    routed_experts = {
        int(expert): decode_real_expert_weights(tensors, config=config, layer=layer, expert=int(expert))
        for expert in selected_experts
    }
    prefix = f"layers.{layer}"
    return TraceableDecodeWeights(
        attention=attention,
        kv=kv,
        attn_norm=tensors[f"layers.{layer}.attn_norm.weight"].contiguous().to(torch.bfloat16),
        ffn_norm=tensors[ffn_norm_key].contiguous().to(torch.bfloat16),
        router_gate=tensors[f"{prefix}.ffn.gate.weight"].contiguous().to(torch.bfloat16),
        router_bias=(
            tensors[f"{prefix}.ffn.gate.bias"].contiguous().to(torch.bfloat16)
            if f"{prefix}.ffn.gate.bias" in tensors
            else None
        ),
        router_tid2eid=(
            tensors[f"{prefix}.ffn.gate.tid2eid"].contiguous().to(torch.long)
            if f"{prefix}.ffn.gate.tid2eid" in tensors
            else None
        ),
        shared_expert=shared_expert,
        routed_experts=routed_experts,
    )


def build_torch_traceable_decode_subpath_reference(
    weights: TraceableDecodeWeights,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    activation: torch.Tensor,
    kv_cache_initial: torch.Tensor,
    cache_len: int,
    cache_update_index: int,
    attention_cache_window_index: int | None = None,
    rope_position_index: int | None = None,
    attention_mode: str = DEFAULT_TRACEABLE_DECODE_ATTENTION_MODE,
    route_plan: TraceableDecodeRoutePlan | None = None,
) -> dict[str, torch.Tensor]:
    _validate_activation(activation, hidden_size=int(config.hidden_size))
    _validate_kv_cache_initial(kv_cache_initial, cache_len=cache_len, kv_output_dim=_kv_output_dim(config))
    attn_norm_output = rms_norm(
        activation[:, 0],
        weights.attn_norm,
        eps=float(config.rms_norm_eps),
    ).unsqueeze(1)
    q_rank_linear = F.linear(attn_norm_output[:, 0].float(), weights.attention.wq_a.float()).to(torch.bfloat16)
    q_rank_norm = rms_norm(q_rank_linear, weights.attention.q_norm, eps=float(config.rms_norm_eps)).unsqueeze(1)
    q_output = F.linear(q_rank_norm[:, 0].float(), weights.attention.wq_b.float()).unsqueeze(1).to(torch.bfloat16)
    kv_linear = F.linear(attn_norm_output[:, 0].float(), weights.kv.wkv.float()).to(torch.bfloat16)
    kv_output = rms_norm(kv_linear, weights.kv.kv_norm, eps=float(config.rms_norm_eps)).unsqueeze(1)
    kv_cache = kv_cache_initial.clone().to(torch.bfloat16)
    kv_cache[:, :, int(cache_update_index) : int(cache_update_index) + 1, :] = kv_output[:, :, :1, :]
    attention_cache_window_index = (
        int(cache_update_index) if attention_cache_window_index is None else int(attention_cache_window_index)
    )
    attention_cache_window = kv_cache[
        :, :, attention_cache_window_index : attention_cache_window_index + int(activation.shape[-2]), :
    ].contiguous()
    rope_position_index = int(attention_cache_window_index) if rope_position_index is None else int(rope_position_index)
    attention_intermediates = _torch_fixed_window_attention(
        q_output=q_output,
        attention_cache_window=attention_cache_window,
        attention_mode=attention_mode,
        config=config,
        layer=layer,
        start_pos=rope_position_index,
    )
    rope_cos, rope_sin = _torch_rope_cos_sin(
        config,
        layer=layer,
        start_pos=rope_position_index,
        seq_len=int(activation.shape[-2]),
    )
    attention_output = attention_intermediates["attention_output"]

    if weights.attention.wo_a is None or weights.attention.wo_b is None:
        raise ValueError("Traceable decode reference requires output projection weights")
    output_rank = grouped_output_projection_a(
        attention_output[:, 0], weights.attention.wo_a, o_groups=int(config.o_groups)
    )
    attention_projected = F.linear(output_rank.float(), weights.attention.wo_b.float()).unsqueeze(1).to(torch.bfloat16)
    post_attention_residual = (activation.float() + attention_projected.float()).to(torch.bfloat16)
    ffn_norm_output = rms_norm(
        post_attention_residual[:, 0],
        weights.ffn_norm,
        eps=float(config.rms_norm_eps),
    ).unsqueeze(1)
    router_intermediates = _torch_router_trace(
        ffn_norm_output,
        weights,
        config=config,
        route_plan=route_plan,
    )
    shared_output = (
        swiglu_expert(
            ffn_norm_output[:, 0].reshape(-1, int(config.hidden_size)),
            weights.shared_expert["w1"],
            weights.shared_expert["w2"],
            weights.shared_expert["w3"],
            swiglu_limit=float(config.swiglu_limit),
        )
        .reshape(activation.shape[0], activation.shape[-2], int(config.hidden_size))
        .unsqueeze(1)
    )
    if route_plan is None:
        routed_output = torch.zeros_like(shared_output)
        routed_expert_outputs: dict[str, torch.Tensor] = {}
    else:
        routed_output_float = torch.zeros_like(shared_output, dtype=torch.float32)
        routed_expert_outputs = {}
        router_selected_weights = router_intermediates["router_selected_route_weights_masked"][0, 0].float()
        for slot, expert in enumerate(route_plan.selected_expert_ids):
            if expert not in weights.routed_experts:
                raise KeyError(f"Missing routed expert weights for selected expert {expert}")
            expert_weights = weights.routed_experts[expert]
            route_weight = router_selected_weights[:, int(slot) : int(slot) + 1]
            expert_output = (
                swiglu_expert(
                    ffn_norm_output[:, 0].reshape(-1, int(config.hidden_size)),
                    expert_weights["w1"],
                    expert_weights["w2"],
                    expert_weights["w3"],
                    route_weight=route_weight,
                    swiglu_limit=float(config.swiglu_limit),
                )
                .reshape(activation.shape[0], activation.shape[-2], int(config.hidden_size))
                .unsqueeze(1)
                .to(torch.bfloat16)
            )
            routed_expert_outputs[f"routed_expert_{expert}_output"] = expert_output
            routed_output_float += expert_output.float()
        routed_output = routed_output_float.to(torch.bfloat16)
    combined_ffn_output = (shared_output.float() + routed_output.float()).to(torch.bfloat16)
    residual_output = (post_attention_residual.float() + combined_ffn_output.float()).to(torch.bfloat16)
    return {
        "attn_norm_output": attn_norm_output.to(torch.bfloat16),
        "q_rank_norm": q_rank_norm.to(torch.bfloat16),
        "q_output": q_output.to(torch.bfloat16),
        "kv_linear": kv_linear.unsqueeze(1).to(torch.bfloat16),
        "kv_output": kv_output.to(torch.bfloat16),
        "kv_cache": kv_cache,
        "attention_cache_window": attention_cache_window,
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        **attention_intermediates,
        "attention_projected": attention_projected.to(torch.bfloat16),
        "post_attention_residual": post_attention_residual.to(torch.bfloat16),
        "ffn_norm_output": ffn_norm_output.to(torch.bfloat16),
        **router_intermediates,
        "shared_output": shared_output.to(torch.bfloat16),
        **routed_expert_outputs,
        "routed_output": routed_output.to(torch.bfloat16),
        "combined_ffn_output": combined_ffn_output.to(torch.bfloat16),
        "residual_output": residual_output,
    }


def build_traceable_decode_route_plan(
    weights: TraceableDecodeWeights,
    *,
    config: DeepSeekV4FlashConfig,
    seq_len: int,
    ffn_norm_output: torch.Tensor,
    routed_topk_prefix: int | None,
    decode_token_index: int = 0,
) -> TraceableDecodeRoutePlan:
    _validate_activation(ffn_norm_output, hidden_size=int(config.hidden_size))
    if int(ffn_norm_output.shape[-2]) != int(seq_len):
        raise ValueError(f"ffn_norm_output token count must be {seq_len}, got {ffn_norm_output.shape[-2]}")
    if not 0 <= int(decode_token_index) < int(seq_len):
        raise ValueError(f"decode_token_index must be in [0, {seq_len}), got {decode_token_index}")
    full_topk = int(config.num_experts_per_tok)
    topk_prefix = full_topk if routed_topk_prefix is None else int(routed_topk_prefix)
    if topk_prefix <= 0:
        raise ValueError(f"routed_topk_prefix must be positive when provided, got {routed_topk_prefix}")
    if topk_prefix > full_topk:
        raise ValueError(f"routed_topk_prefix {topk_prefix} exceeds num_experts_per_tok {full_topk}")

    input_ids = _traceable_decode_input_ids(weights.router_tid2eid, seq_len=seq_len)
    token_input_ids = None if input_ids is None else input_ids[:, int(decode_token_index) : int(decode_token_index) + 1]
    token = ffn_norm_output[:, 0, int(decode_token_index) : int(decode_token_index) + 1, :]
    router_weights, router_indices = v4_router(
        token,
        weights.router_gate,
        topk=full_topk,
        route_scale=float(config.routed_scaling_factor),
        scoring_func=str(config.scoring_func),
        bias=weights.router_bias,
        input_ids=token_input_ids,
        tid2eid=weights.router_tid2eid,
    )
    selected_weights = router_weights[..., :topk_prefix].contiguous().to(torch.bfloat16)
    selected_indices = router_indices[..., :topk_prefix].contiguous().to(torch.long)
    selected_ids = [int(value) for value in selected_indices.reshape(-1).tolist()]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError(f"Traceable decode route prefix selected duplicate experts: {selected_ids}")

    per_expert_route_weight: dict[int, torch.Tensor] = {}
    for slot, expert in enumerate(selected_ids):
        route_weight = torch.zeros((1, int(seq_len), 1), dtype=torch.bfloat16)
        route_weight[0, int(decode_token_index), 0] = selected_weights[0, 0, slot]
        per_expert_route_weight[int(expert)] = route_weight.contiguous()

    return TraceableDecodeRoutePlan(
        decode_token_index=int(decode_token_index),
        topk_prefix=topk_prefix,
        full_topk=full_topk,
        router_weights=router_weights.contiguous(),
        router_indices=router_indices.contiguous().to(torch.long),
        selected_weights=selected_weights,
        selected_indices=selected_indices,
        input_ids=input_ids,
        per_expert_route_weight=per_expert_route_weight,
    )


def _torch_router_trace(
    ffn_norm_output: torch.Tensor,
    weights: TraceableDecodeWeights,
    *,
    config: DeepSeekV4FlashConfig,
    route_plan: TraceableDecodeRoutePlan | None,
) -> dict[str, torch.Tensor]:
    _validate_activation(ffn_norm_output, hidden_size=int(config.hidden_size))
    router_logits = torch.matmul(ffn_norm_output.float(), weights.router_gate.float().T)
    if str(config.scoring_func) == "softmax":
        router_scores = router_logits.softmax(dim=-1)
    elif str(config.scoring_func) == "sigmoid":
        router_scores = router_logits.sigmoid()
    elif str(config.scoring_func) == "sqrtsoftplus":
        router_scores = F.softplus(router_logits).sqrt()
    else:
        raise ValueError(f"Unsupported DeepSeek V4 Flash scoring_func {config.scoring_func!r}")

    router_selection_scores = (
        router_scores if weights.router_bias is None else router_scores + weights.router_bias.float().view(1, 1, 1, -1)
    )
    input_ids = (
        _traceable_decode_input_ids(weights.router_tid2eid, seq_len=int(ffn_norm_output.shape[-2]))
        if route_plan is None
        else route_plan.input_ids
    )
    if weights.router_tid2eid is not None:
        if input_ids is None:
            raise ValueError("input_ids is required for traceable decode hash router reference")
        router_topk_indices = weights.router_tid2eid[input_ids.reshape(-1)].to(torch.long)
        router_topk_indices = router_topk_indices.reshape(
            int(ffn_norm_output.shape[0]),
            1,
            int(ffn_norm_output.shape[-2]),
            int(config.num_experts_per_tok),
        )
        router_topk_selection_scores = router_selection_scores.gather(-1, router_topk_indices)
    else:
        router_topk_selection_scores, router_topk_indices = router_selection_scores.topk(
            int(config.num_experts_per_tok),
            dim=-1,
        )

    if weights.router_bias is None and weights.router_tid2eid is None:
        router_topk_route_scores = router_topk_selection_scores
    else:
        router_topk_route_scores = router_scores.gather(-1, router_topk_indices)

    if str(config.scoring_func) == "softmax":
        router_route_weights = router_topk_route_scores * float(config.routed_scaling_factor)
    else:
        router_route_weights = router_topk_route_scores / (router_topk_route_scores.sum(dim=-1, keepdim=True) + 1e-20)
        router_route_weights = router_route_weights * float(config.routed_scaling_factor)

    topk_prefix = int(config.num_experts_per_tok) if route_plan is None else int(route_plan.topk_prefix)
    decode_token_index = 0 if route_plan is None else int(route_plan.decode_token_index)
    router_selected_route_weights = router_route_weights[..., :topk_prefix].contiguous()
    row_mask = torch.zeros_like(router_selected_route_weights[..., :1])
    row_mask[..., decode_token_index : decode_token_index + 1, :] = 1.0
    router_selected_route_weights_masked = router_selected_route_weights * row_mask
    router_decode_topk_indices = router_topk_indices[:, :, decode_token_index : decode_token_index + 1, :].contiguous()
    router_decode_route_weights = router_route_weights[
        :, :, decode_token_index : decode_token_index + 1, :
    ].contiguous()
    router_decode_selected_route_weights = router_selected_route_weights[
        :, :, decode_token_index : decode_token_index + 1, :
    ].contiguous()
    return {
        "router_logits": router_logits.to(torch.bfloat16),
        "router_scores": router_scores.to(torch.bfloat16),
        "router_selection_scores": router_selection_scores.to(torch.bfloat16),
        "router_topk_selection_scores": router_topk_selection_scores.to(torch.bfloat16),
        "router_topk_indices": router_topk_indices.to(torch.uint16),
        "router_topk_route_scores": router_topk_route_scores.to(torch.bfloat16),
        "router_route_weights": router_route_weights.to(torch.bfloat16),
        "router_selected_route_weights": router_selected_route_weights.to(torch.bfloat16),
        "router_selected_route_weights_masked": router_selected_route_weights_masked.to(torch.bfloat16),
        "router_decode_topk_indices": router_decode_topk_indices.to(torch.uint16),
        "router_decode_route_weights": router_decode_route_weights.to(torch.bfloat16),
        "router_decode_selected_route_weights": router_decode_selected_route_weights.to(torch.bfloat16),
    }


def default_guarded_symbols() -> tuple[GuardedSymbol, ...]:
    return (
        GuardedSymbol("ttnn", "to_torch", "ttnn.to_torch"),
        GuardedSymbol("ttnn", "from_torch", "ttnn.from_torch"),
        GuardedSymbol("ttnn", "from_device", "ttnn.from_device"),
        GuardedSymbol("ttnn", "copy_host_to_device_tensor", "ttnn.copy_host_to_device_tensor"),
        GuardedSymbol("ttnn", "copy_host_to_device_tensor_partial", "ttnn.copy_host_to_device_tensor_partial"),
        GuardedSymbol(
            "models.demos.deepseek_v4_flash.ttnn_attention_projection",
            "_ttnn_projection_to_torch_3d",
            "TtAttentionProjection._ttnn_projection_to_torch_3d",
        ),
        GuardedSymbol(
            "models.demos.deepseek_v4_flash.ttnn_attention_projection",
            "grouped_output_projection_a",
            "grouped_output_projection_a(host_wo_a)",
        ),
        GuardedSymbol(
            "models.demos.deepseek_v4_flash.ttnn_sparse_attention",
            "TtSparsePrefillAttention.forward",
            "TtSparsePrefillAttention.forward(host_sparse_attention)",
        ),
        GuardedSymbol(
            "models.demos.deepseek_v4_flash.ttnn_router",
            "TtRouter.forward",
            "TtRouter.forward(host_topk)",
        ),
        GuardedSymbol(
            "models.demos.deepseek_v4_flash.ttnn_router",
            "select_router_scores",
            "select_router_scores(torch_topk_or_hash)",
        ),
        GuardedSymbol(
            "models.demos.deepseek_v4_flash.ttnn_moe_block",
            "TtMoEFeedForwardBlock.forward",
            "TtMoEFeedForwardBlock.forward(host_expert_plan)",
        ),
        GuardedSymbol(
            "models.demos.deepseek_v4_flash.ttnn_expert_group",
            "TtPlannedRoutedExpertGroup.run_torch_host_combine",
            "TtPlannedRoutedExpertGroup.run_torch_host_combine",
        ),
        GuardedSymbol(
            "models.demos.deepseek_v4_flash.ttnn_prefill_indexer",
            "TtPrefillIndexer.forward",
            "TtPrefillIndexer.forward(host_indexer_topk)",
        ),
        GuardedSymbol(
            "models.demos.deepseek_v4_flash.ttnn_prefill_indexer",
            "TtPrefillIndexer.topk_from_q_rank",
            "TtPrefillIndexer.topk_from_q_rank(host_indexer_topk)",
        ),
        GuardedSymbol(
            "models.demos.deepseek_v4_flash.ttnn_prefill_indexer",
            "TtPrefillIndexer.topk_from_q_rank_and_cache",
            "TtPrefillIndexer.topk_from_q_rank_and_cache(host_indexer_topk)",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace/replay the first TTNN-only DeepSeek V4 Flash decode subpath.")
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=DEFAULT_TRACEABLE_DECODE_LAYER)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_TRACEABLE_DECODE_SEQ_LEN)
    parser.add_argument(
        "--routed-topk-prefix",
        type=int,
        default=DEFAULT_TRACEABLE_DECODE_ROUTED_TOPK_PREFIX,
        help="Optional routed expert prefix limit; omit to trace the full configured top-k fanout.",
    )
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_TRACEABLE_DECODE_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_TRACEABLE_DECODE_MAX_BYTES)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--trace-region-size", type=int, default=DEFAULT_TRACEABLE_DECODE_TRACE_REGION_SIZE)
    parser.add_argument("--cache-len", type=int, default=DEFAULT_TRACEABLE_DECODE_CACHE_LEN)
    parser.add_argument("--cache-update-index", type=int, default=None)
    parser.add_argument("--decode-steps", type=int, default=DEFAULT_TRACEABLE_DECODE_STEPS)
    parser.add_argument(
        "--attention-mode",
        choices=TRACEABLE_DECODE_ATTENTION_MODES,
        default=DEFAULT_TRACEABLE_DECODE_ATTENTION_MODE,
        help="Traceable fixed-window attention implementation to use; qk-softmax is the default.",
    )
    parser.add_argument(
        "--cache-update-api",
        choices=TRACEABLE_DECODE_CACHE_UPDATE_APIS,
        default=DEFAULT_TRACEABLE_DECODE_CACHE_UPDATE_API,
        help=(
            "Cache write primitive for the protected decode body. Use paged_update_cache to replay one trace while "
            "mutating update_idxs_tensor outside the guard."
        ),
    )
    parser.add_argument(
        "--rope-position-api",
        choices=TRACEABLE_DECODE_ROPE_POSITION_APIS,
        default=DEFAULT_TRACEABLE_DECODE_ROPE_POSITION_API,
        help=(
            "RoPE cos/sin selection primitive for the protected decode body. Use embedding_position_tensor to "
            "gather fixed-shape cos/sin rows from a mutable device position tensor during trace replay."
        ),
    )
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--pcc", type=float, default=0.99)
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--verbose-logs", action="store_true")
    args = parser.parse_args()

    if not args.verbose_logs:
        os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
        os.environ.setdefault("LOGGER_LEVEL", "Warning")

    result = run_traceable_decode_subpath_smoke(
        args.snapshot_dir,
        layer=args.layer,
        seq_len=args.seq_len,
        routed_topk_prefix=args.routed_topk_prefix,
        max_tensors=args.max_tensors,
        max_bytes=args.max_bytes,
        cpu_only=args.cpu_only,
        device_id=args.device_id,
        trace_region_size=args.trace_region_size,
        cache_len=args.cache_len,
        cache_update_index=args.cache_update_index,
        decode_steps=args.decode_steps,
        attention_mode=args.attention_mode,
        cache_update_api=args.cache_update_api,
        rope_position_api=args.rope_position_api,
        pcc=args.pcc,
        rtol=args.rtol,
        atol=args.atol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def _run_ttnn_traceable_decode_subpath(
    weights: TraceableDecodeWeights,
    *,
    route_plans: Sequence[TraceableDecodeRoutePlan],
    config: DeepSeekV4FlashConfig,
    activations: Sequence[torch.Tensor],
    replay_activations: Sequence[torch.Tensor],
    kv_cache_initial: torch.Tensor,
    device_id: int,
    trace_region_size: int,
    layer: int,
    cache_len: int,
    cache_update_indices: Sequence[int],
    attention_mode: str,
    cache_update_api: str,
    rope_position_api: str,
    attention_window_index: int,
    rope_position_indices: Sequence[int],
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    if not route_plans:
        raise ValueError("at least one route plan is required")
    if not (len(route_plans) == len(activations) == len(replay_activations) == len(cache_update_indices)):
        raise ValueError(
            "route_plans, activations, replay_activations, and cache_update_indices must have matching lengths"
        )
    cache_update_api = _validate_cache_update_api(cache_update_api)
    rope_position_api = _validate_rope_position_api(rope_position_api)
    if len(rope_position_indices) != len(cache_update_indices):
        raise ValueError("rope_position_indices and cache_update_indices must have matching lengths")
    device = ttnn.open_device(
        device_id=int(device_id),
        num_command_queues=1,
        trace_region_size=int(trace_region_size),
    )
    trace_ids: list[int] = []
    try:
        allocated_trace_count = 0
        kv_cache = _to_tt_kv_cache(
            cache_len=cache_len,
            kv_output_dim=_kv_output_dim(config),
            initial_cache=kv_cache_initial,
            device=device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_input = ttnn.allocate_tensor_on_device(
            ttnn.Shape(tuple(activations[0].shape)),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
        )
        outputs_by_step: list[dict[str, torch.Tensor]] = []
        guarded_labels: list[str] = []
        if cache_update_api == TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR:
            update_idxs_tensor = _to_tt_cache_update_idxs_tensor(
                int(cache_update_indices[0]),
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            rope_position_idxs_tensor = (
                _to_tt_rope_position_idxs_tensor(
                    int(rope_position_indices[0]),
                    seq_len=int(activations[0].shape[-2]),
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR
                else None
            )
            module = TtTraceableDecodeSubpath(
                device=device,
                weights=weights,
                route_plan=route_plans[0],
                config=config,
                layer=layer,
                cache_len=cache_len,
                cache_update_index=int(cache_update_indices[0]),
                cache_update_api=cache_update_api,
                cache_update_idxs_tensor=update_idxs_tensor,
                rope_position_api=rope_position_api,
                rope_position_idxs_tensor=rope_position_idxs_tensor,
                attention_cache_window_index=int(attention_window_index),
                rope_position_index=int(rope_position_indices[0]),
                initial_kv_cache=None,
                kv_cache=kv_cache,
                attention_mode=attention_mode,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            _copy_activation_to_device(activations[0], tt_input)
            _copy_cache_update_idx_to_device(int(cache_update_indices[0]), update_idxs_tensor)
            if rope_position_idxs_tensor is not None:
                _copy_rope_position_idxs_to_device(
                    int(rope_position_indices[0]),
                    seq_len=int(activations[0].shape[-2]),
                    tt_rope_position_idxs=rope_position_idxs_tensor,
                )
            module(tt_input)
            ttnn.synchronize_device(device)
            kv_cache = module.kv_cache

            with TraceableDecodeHostGuard() as guard:
                trace_id = ttnn.begin_trace_capture(device, cq_id=0)
                output_tensors = module(tt_input)
                ttnn.end_trace_capture(device, trace_id, cq_id=0)
            trace_ids.append(trace_id)
            allocated_trace_count += 1
            guarded_labels = guard.guarded_labels
            kv_cache = module.kv_cache
            ttnn.synchronize_device(device)

            for replay_activation, cache_update_index, rope_position_index in zip(
                replay_activations, cache_update_indices, rope_position_indices
            ):
                _copy_activation_to_device(replay_activation, tt_input)
                _copy_cache_update_idx_to_device(int(cache_update_index), update_idxs_tensor)
                if rope_position_idxs_tensor is not None:
                    _copy_rope_position_idxs_to_device(
                        int(rope_position_index),
                        seq_len=int(replay_activation.shape[-2]),
                        tt_rope_position_idxs=rope_position_idxs_tensor,
                    )
                ttnn.synchronize_device(device)
                ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
                ttnn.synchronize_device(device)
                outputs_by_step.append(
                    {name: ttnn.to_torch(tensor).contiguous() for name, tensor in output_tensors.items()}
                )
                kv_cache = module.kv_cache

            ttnn.release_trace(device, trace_id)
            trace_ids.remove(trace_id)
            trace_info = {
                "attempted": True,
                "capture_passed": True,
                "execute_replay_attempted": True,
                "execute_replay_passed": True,
                "trace_id_allocated": True,
                "trace_ids_allocated": allocated_trace_count,
                "guard_enabled": True,
                "guarded_symbols": guarded_labels,
                "ttnn_to_torch_guarded": "ttnn.to_torch" in guarded_labels,
                "host_boundaries_inside_trace": [],
                "decode_step_count": len(route_plans),
                "capture_count": allocated_trace_count,
                "positions_used": [int(value) for value in cache_update_indices],
                "one_trace_capture_replayed_across_positions": len(route_plans) > 1,
                "single_capture_replayed_across_positions": len(route_plans) > 1,
                "recaptured_per_position": False,
                "carried_device_kv_cache_state": True,
                "cache_update_api": cache_update_api,
                "rope_position_api": rope_position_api,
                "update_index_source": "device_tensor",
                "cache_update_index_dynamic": True,
                "cache_update_index_kind": "replay_mutable_device_tensor",
                "cache_read_window_dynamic": False,
                "cache_read_window_status": "static_single_capture_initial_position",
                "rope_position_dynamic": rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR,
                "rope_position_kind": "replay_mutable_device_tensor"
                if rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR
                else "static_host_materialized_tables",
                "rope_position_status": "replay_mutable_device_tensor_embedding"
                if rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR
                else "static_single_capture_initial_position",
                "rope_positions_used": [int(value) for value in rope_position_indices],
            }
            return outputs_by_step, trace_info

        for step, (route_plan, activation, replay_activation, cache_update_index, rope_position_index) in enumerate(
            zip(route_plans, activations, replay_activations, cache_update_indices, rope_position_indices)
        ):
            rope_position_idxs_tensor = (
                _to_tt_rope_position_idxs_tensor(
                    int(rope_position_index),
                    seq_len=int(activation.shape[-2]),
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR
                else None
            )
            module = TtTraceableDecodeSubpath(
                device=device,
                weights=weights,
                route_plan=route_plan,
                config=config,
                layer=layer,
                cache_len=cache_len,
                cache_update_index=int(cache_update_index),
                cache_update_api=cache_update_api,
                rope_position_api=rope_position_api,
                rope_position_idxs_tensor=rope_position_idxs_tensor,
                attention_cache_window_index=int(cache_update_index),
                rope_position_index=int(rope_position_index),
                initial_kv_cache=None,
                kv_cache=kv_cache,
                attention_mode=attention_mode,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            _copy_activation_to_device(activation, tt_input)
            if rope_position_idxs_tensor is not None:
                _copy_rope_position_idxs_to_device(
                    int(rope_position_index),
                    seq_len=int(activation.shape[-2]),
                    tt_rope_position_idxs=rope_position_idxs_tensor,
                )
            module(tt_input)
            ttnn.synchronize_device(device)
            kv_cache = module.kv_cache

            with TraceableDecodeHostGuard() as guard:
                trace_id = ttnn.begin_trace_capture(device, cq_id=0)
                output_tensors = module(tt_input)
                ttnn.end_trace_capture(device, trace_id, cq_id=0)
            trace_ids.append(trace_id)
            allocated_trace_count += 1
            guarded_labels = guard.guarded_labels
            kv_cache = module.kv_cache
            ttnn.synchronize_device(device)

            _copy_activation_to_device(replay_activation, tt_input)
            if rope_position_idxs_tensor is not None:
                _copy_rope_position_idxs_to_device(
                    int(rope_position_index),
                    seq_len=int(replay_activation.shape[-2]),
                    tt_rope_position_idxs=rope_position_idxs_tensor,
                )
            ttnn.synchronize_device(device)
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(device)
            outputs_by_step.append(
                {name: ttnn.to_torch(tensor).contiguous() for name, tensor in output_tensors.items()}
            )
            ttnn.release_trace(device, trace_id)
            trace_ids.remove(trace_id)
            kv_cache = module.kv_cache

        trace_info = {
            "attempted": True,
            "capture_passed": True,
            "execute_replay_attempted": True,
            "execute_replay_passed": True,
            "trace_id_allocated": True,
            "trace_ids_allocated": allocated_trace_count,
            "guard_enabled": True,
            "guarded_symbols": guarded_labels,
            "ttnn_to_torch_guarded": "ttnn.to_torch" in guarded_labels,
            "host_boundaries_inside_trace": [],
            "decode_step_count": len(route_plans),
            "capture_count": allocated_trace_count,
            "positions_used": [int(value) for value in cache_update_indices],
            "one_trace_capture_replayed_across_positions": False,
            "single_capture_replayed_across_positions": False,
            "recaptured_per_position": len(route_plans) > 1,
            "carried_device_kv_cache_state": True,
            "cache_update_api": cache_update_api,
            "rope_position_api": rope_position_api,
            "update_index_source": "host_scalar",
            "cache_update_index_dynamic": False,
            "cache_update_index_kind": "static_host_argument_per_trace_capture",
            "cache_read_window_dynamic": False,
            "cache_read_window_status": "static_per_trace_capture",
            "rope_position_dynamic": rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR,
            "rope_position_kind": "replay_mutable_device_tensor"
            if rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR
            else "static_host_materialized_tables",
            "rope_position_status": "replay_mutable_device_tensor_embedding_per_trace_capture"
            if rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR
            else "static_per_trace_capture",
            "rope_positions_used": [int(value) for value in rope_position_indices],
        }
        return outputs_by_step, trace_info
    finally:
        for trace_id in reversed(trace_ids):
            ttnn.release_trace(device, trace_id)
        ttnn.close_device(device)


def _copy_activation_to_device(activation: torch.Tensor, tt_input) -> None:
    host_tensor = ttnn.from_torch(
        activation.contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn.copy_host_to_device_tensor(host_tensor, tt_input)


def _to_tt_cache_update_idxs_tensor(
    cache_update_index: int,
    *,
    device,
    memory_config,
):
    return ttnn.from_torch(
        torch.tensor([int(cache_update_index)], dtype=torch.int32),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,
    )


def _copy_cache_update_idx_to_device(cache_update_index: int, tt_update_idxs) -> None:
    host_tensor = ttnn.from_torch(
        torch.tensor([int(cache_update_index)], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn.copy_host_to_device_tensor(host_tensor, tt_update_idxs)


def _to_tt_rope_position_idxs_tensor(
    start_pos: int,
    *,
    seq_len: int,
    device,
    memory_config,
):
    return ttnn.from_torch(
        _rope_position_idx_values(start_pos, seq_len=seq_len),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,
    )


def _copy_rope_position_idxs_to_device(start_pos: int, *, seq_len: int, tt_rope_position_idxs) -> None:
    host_tensor = ttnn.from_torch(
        _rope_position_idx_values(start_pos, seq_len=seq_len),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn.copy_host_to_device_tensor(host_tensor, tt_rope_position_idxs)


def _rope_position_idx_values(start_pos: int, *, seq_len: int) -> torch.Tensor:
    return torch.arange(int(start_pos), int(start_pos) + int(seq_len), dtype=torch.int32).reshape(1, int(seq_len))


def _base_result(
    *,
    snapshot_dir: Path,
    layer: int,
    seq_len: int,
    cache_len: int,
    cache_update_index: int,
    cache_update_indices: Sequence[int],
    decode_steps: int,
    config: DeepSeekV4FlashConfig,
    metadata: Sequence[TensorMetadata],
    metadata_groups: Mapping[str, Sequence[TensorMetadata]],
    weights: TraceableDecodeWeights,
    activation: torch.Tensor,
    kv_cache_initial: torch.Tensor,
    replay_activation: torch.Tensor,
    activations: Sequence[torch.Tensor],
    replay_activations: Sequence[torch.Tensor],
    route_plans: Sequence[TraceableDecodeRoutePlan],
    reference: Mapping[str, torch.Tensor],
    replay_reference: Mapping[str, torch.Tensor],
    references: Sequence[Mapping[str, torch.Tensor]],
    replay_references: Sequence[Mapping[str, torch.Tensor]],
    attention_mode: str,
    cache_update_api: str,
    rope_position_api: str,
    attention_window_indices: Sequence[int],
    rope_position_indices: Sequence[int],
    max_tensors: int,
    max_bytes: int,
    trace_region_size: int,
) -> dict[str, Any]:
    loaded_groups = {
        name: {
            "count": len(items),
            "payload_bytes": sum(item.nbytes for item in items),
            "canonical_keys": [item.canonical_key for item in items],
        }
        for name, items in metadata_groups.items()
    }
    guarded_labels = [symbol.label for symbol in default_guarded_symbols()]
    route_plan = route_plans[0]
    cache_update_api = _validate_cache_update_api(cache_update_api)
    rope_position_api = _validate_rope_position_api(rope_position_api)
    uses_device_update_index = cache_update_api == TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR
    uses_device_rope_position = rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR
    update_index_source = "device_tensor" if uses_device_update_index else "host_scalar"
    update_index_kind = (
        "replay_mutable_device_tensor" if uses_device_update_index else "static_host_argument_per_trace_capture"
    )
    cache_read_window_status = (
        "static_single_capture_initial_position" if uses_device_update_index else "static_per_trace_capture"
    )
    rope_position_status = (
        "replay_mutable_device_tensor_embedding"
        if uses_device_rope_position
        else ("static_single_capture_initial_position" if uses_device_update_index else "static_per_trace_capture")
    )
    rope_position_kind = (
        "replay_mutable_device_tensor" if uses_device_rope_position else "static_host_materialized_tables"
    )
    routed_expert_ids_loaded = [int(expert) for expert in weights.routed_experts]
    routed_expert_ids_executed = _unique_ints(
        expert for step_route_plan in route_plans for expert in step_route_plan.selected_expert_ids
    )
    router_summary = _router_trace_summary(route_plan, config=config, weights=weights)
    excluded_from_trace = [
        "true DeepSeek K/V split; this slice creates explicit K and V tensors from one compressed cache window",
        "dynamic sparse indexer top-k and per-token cache gather",
        "DeepSeek sparse attention-sink/indexer semantics; fixed-window dense softmax uses contiguous cache rows",
        "dynamic MoE expert dispatch; selected expert modules are statically instantiated from a host preflight plan",
        "dynamic cache read-window advancement beyond the fixed traced window",
        "embedding and logits",
    ]
    if not uses_device_rope_position:
        excluded_from_trace.insert(4, "dynamic RoPE position advancement beyond the fixed traced window")
    if not uses_device_update_index:
        excluded_from_trace.insert(4, "cache write advancement beyond the fixed traced update index")
    if not router_summary["topk_in_trace"]:
        excluded_from_trace.insert(
            4,
            "router top-k for hash-routed layers; tid2eid metadata is still materialized as static device indices",
        )
    if attention_mode == TRACEABLE_DECODE_ATTENTION_LEGACY_BLEND_MODE:
        excluded_from_trace.insert(3, "QK scoring, softmax, and value reduction in legacy q+kv blend mode")
        excluded_from_trace.insert(4, "Q/K RoPE split in legacy q+kv blend mode")
    if route_plan.topk_prefix < route_plan.full_topk:
        excluded_from_trace.insert(
            4,
            "routed experts outside the configured top-k prefix when topk_prefix_limit is less than full top-k",
        )
    return {
        "schema_version": REAL_TRACEABLE_DECODE_SMOKE_SCHEMA_VERSION,
        "mode": "unexecuted",
        "snapshot_dir": str(snapshot_dir),
        "layer": int(layer),
        "logical_decode_tokens": 1,
        "decode_step_count": int(decode_steps),
        "positions": [int(value) for value in cache_update_indices],
        "positions_used": [int(value) for value in cache_update_indices],
        "decode_positions": [int(value) for value in cache_update_indices],
        "one_trace_capture_replayed_across_positions": False,
        "cache_update_api": cache_update_api,
        "rope_position_api": rope_position_api,
        "update_index_source": update_index_source,
        "per_step_cache_rows_updated": [[int(value)] for value in cache_update_indices],
        "cache_read_window_dynamic": False,
        "cache_read_window_status": cache_read_window_status,
        "rope_position_dynamic": uses_device_rope_position,
        "rope_position_status": rope_position_status,
        "rope_position_kind": rope_position_kind,
        "rope_positions": [int(value) for value in rope_position_indices],
        "tensor_sequence_length": int(seq_len),
        "cache_update": {
            "name": "compressed_kv_projection_cache_append",
            "cache_update_api": cache_update_api,
            "update_index_source": update_index_source,
            "cache_len": int(cache_len),
            "update_index": int(cache_update_index),
            "update_indices": [int(value) for value in cache_update_indices],
            "updated_rows": [int(value) for value in cache_update_indices],
            "per_step_updated_rows": [[int(value)] for value in cache_update_indices],
            "updated_tokens": 1,
            "updated_tokens_per_step": 1,
            "decode_steps": int(decode_steps),
            "update_index_kind": update_index_kind,
            "dynamic_update_index_in_trace": uses_device_update_index,
            "single_capture_replay_across_positions": False,
            "input_layout": "[seq=1, heads=1, batch_padded=32, kv_output_dim]",
            "cache_layout": "[batch=1, heads=1, cache_len, kv_output_dim]",
            "device_resident_inside_trace": True,
            "cache_read_window_dynamic": False,
            "cache_read_window_status": cache_read_window_status,
            "rope_position_dynamic": uses_device_rope_position,
            "rope_position_api": rope_position_api,
            "rope_position_kind": rope_position_kind,
            "rope_position_status": rope_position_status,
            "limitation": (
                "ttnn.experimental.paged_update_cache reads update_idxs_tensor from device memory, so the cache write "
                "row can change across replay; the cache slice bounds remain static, and RoPE positions are dynamic "
                "only when rope_position_api=embedding_position_tensor"
                if uses_device_update_index
                else "ttnn.update_cache takes update_idx as a Python scalar in this path; the cache slice bounds and "
                "RoPE table positions are also static trace-capture inputs, so one captured trace is not claimed "
                "to advance across decode positions"
            ),
        },
        "multi_position_replay": {
            "requested_decode_steps": int(decode_steps),
            "positions_used": [int(value) for value in cache_update_indices],
            "carried_device_kv_cache_state": True,
            "single_capture_replayed_across_positions": False,
            "recaptured_per_position": int(decode_steps) > 1,
            "identical_guarded_body_per_position": True,
            "cache_update_api": cache_update_api,
            "rope_position_api": rope_position_api,
            "update_index_source": update_index_source,
            "cache_update_index": update_index_kind,
            "cache_update_index_dynamic": uses_device_update_index,
            "current_position_dynamic": False,
            "cache_read_window_dynamic": False,
            "cache_read_window_status": cache_read_window_status,
            "rope_position_dynamic": uses_device_rope_position,
            "rope_position_kind": rope_position_kind,
            "rope_position_status": rope_position_status,
            "rope_positions_used": [int(value) for value in rope_position_indices],
            "strongest_landed_subpiece": (
                "one guarded TTNN body can be captured once and replayed while device tensors advance both cache "
                "write rows and RoPE cos/sin embedding positions; cache read bounds remain static"
                if uses_device_update_index and uses_device_rope_position
                else (
                    "one guarded TTNN body can be captured once and replayed while a device update index tensor "
                    "advances cache write rows"
                    if uses_device_update_index
                    else (
                        "one guarded TTNN body can be captured per static position while reusing the same "
                        "device-resident KV cache tensor across steps"
                    )
                )
            ),
            "single_capture_blockers": [
                *(
                    []
                    if uses_device_update_index
                    else [
                        "ttnn.update_cache exposes update_idx as a host uint32 argument, not a mutable TTNN tensor input"
                    ]
                ),
                "ttnn.slice cache-window start/end are Python shape arguments baked into the traced op sequence",
                *(
                    []
                    if uses_device_rope_position
                    else ["RoPE cos/sin tables are materialized for a fixed start_pos before trace capture"]
                ),
                "static MoE expert dispatch is built from a host preflight route plan",
            ],
        },
        "position_dependent_decode_inventory": _position_dependent_decode_inventory(
            cache_update_api=cache_update_api,
            rope_position_api=rope_position_api,
            config=config,
        ),
        "attention_path": _attention_path_summary(
            config=config,
            layer=layer,
            cache_update_index=int(attention_window_indices[0]),
            seq_len=seq_len,
            attention_mode=attention_mode,
            cache_write_index=cache_update_index,
            cache_update_api=cache_update_api,
            rope_position_index=int(rope_position_indices[0]),
            rope_position_api=rope_position_api,
        ),
        "attention_path_by_step": [
            _attention_path_summary(
                config=config,
                layer=layer,
                cache_update_index=int(attention_window_indices[step]),
                seq_len=seq_len,
                attention_mode=attention_mode,
                cache_write_index=int(cache_update_indices[step]),
                cache_update_api=cache_update_api,
                rope_position_index=int(rope_position_indices[step]),
                rope_position_api=rope_position_api,
            )
            for step in range(len(cache_update_indices))
        ],
        "traceability_flags": {
            "attention_mode": attention_mode,
            "kv_source": "device_resident_compressed_kv_projection_cache_window",
            "kv_split_in_trace": attention_mode == TRACEABLE_DECODE_ATTENTION_QK_SOFTMAX_MODE,
            "true_kv_split_in_trace": False,
            "rope_in_trace": attention_mode == TRACEABLE_DECODE_ATTENTION_QK_SOFTMAX_MODE,
            "qk_scores_in_trace": attention_mode == TRACEABLE_DECODE_ATTENTION_QK_SOFTMAX_MODE,
            "softmax_in_trace": attention_mode == TRACEABLE_DECODE_ATTENTION_QK_SOFTMAX_MODE,
            "context_in_trace": attention_mode == TRACEABLE_DECODE_ATTENTION_QK_SOFTMAX_MODE,
            "host_attention_or_context_input": False,
            "cache_update_api": cache_update_api,
            "cache_update_index_source": update_index_source,
            "cache_write_index_dynamic_in_trace": uses_device_update_index,
            "cache_read_window_dynamic_in_trace": False,
            "rope_position_api": rope_position_api,
            "rope_position_dynamic_in_trace": uses_device_rope_position,
            "router_mode": router_summary["mode"],
            "router_gate_matmul_in_trace": router_summary["gate_matmul_in_trace"],
            "router_scoring_in_trace": router_summary["scoring_in_trace"],
            "router_topk_in_trace": router_summary["topk_in_trace"],
            "router_route_weights_in_trace": router_summary["route_weights_in_trace"],
            "router_indices_dynamic_in_trace": router_summary["indices_dynamic_in_trace"],
            "router_expert_dispatch_dynamic_in_trace": False,
            "router_expert_dispatch_static_in_trace": True,
        },
        "model": {
            "hidden_size": int(config.hidden_size),
            "q_lora_rank": int(config.q_lora_rank),
            "num_attention_heads": int(config.num_attention_heads),
            "num_key_value_heads": int(config.num_key_value_heads),
            "head_dim": int(config.head_dim),
            "q_output_dim": int(config.num_attention_heads) * int(config.head_dim),
            "kv_output_dim": _kv_output_dim(config),
            "o_groups": int(config.o_groups),
            "o_lora_rank": int(config.o_lora_rank),
            "attention_output_rank_dim": int(config.o_groups) * int(config.o_lora_rank),
            "moe_intermediate_size": int(config.moe_intermediate_size),
            "n_routed_experts": int(config.n_routed_experts),
            "num_experts_per_tok": int(config.num_experts_per_tok),
            "scoring_func": str(config.scoring_func),
            "routed_scaling_factor": float(config.routed_scaling_factor),
            "n_shared_experts": int(config.n_shared_experts),
            "shared_intermediate_size": int(config.moe_intermediate_size) * int(config.n_shared_experts),
            "rms_norm_eps": float(config.rms_norm_eps),
            "swiglu_limit": float(config.swiglu_limit),
            "torch_dtype": config.torch_dtype,
            "quantization_config": config.quantization_config,
        },
        "traceable_decode_scope": {
            "name": "traceable_decode_subpath",
            "not_full_forward": True,
            "inside_trace": _traceable_decode_inside_trace_ops(
                attention_mode,
                cache_update_api=cache_update_api,
                config=config,
                weights=weights,
                route_plan=route_plan,
                rope_position_api=rope_position_api,
            ),
            "path": (
                "decode hidden state -> attn_norm/query projection plus K/V projection/cache append; "
                "fixed device cache-window read plus q projection -> grouped wo_a/wo_b/post-attention residual; "
                "post-attention residual -> ffn_norm/device router weights/static routed fanout plus shared expert/residual"
            ),
            "logical_decode_token_policy": (
                "the first token is the logical decode token; tensor shape is tile-padded/static for trace replay"
            ),
            "excluded_from_trace": excluded_from_trace,
            "cache_write_position": "replay_mutable_device_tensor"
            if uses_device_update_index
            else "static_host_scalar",
            "cache_read_window": cache_read_window_status,
            "rope_position": rope_position_status,
            "production_autoregressive_decode": False,
            "production_autoregressive_decode_blocker": (
                "cache write, cache read/current position, and RoPE/attention position are not all dynamic under one "
                "captured trace"
            ),
        },
        "router_trace": router_summary,
        "selected_routing": _route_plan_summary(route_plan, config=config),
        "selected_routing_by_step": [
            _route_plan_summary(step_route_plan, config=config) for step_route_plan in route_plans
        ],
        "routed_expert_execution": {
            "loaded_expert_ids": routed_expert_ids_loaded,
            "loaded_expert_count": len(routed_expert_ids_loaded),
            "executed_expert_ids": routed_expert_ids_executed,
            "executed_expert_count": len(routed_expert_ids_executed),
            "all_selected_experts_loaded": set(routed_expert_ids_executed).issubset(set(routed_expert_ids_loaded)),
            "full_topk_executed": all(
                int(step_route_plan.topk_prefix) == int(step_route_plan.full_topk) for step_route_plan in route_plans
            ),
        },
        "decode_steps_detail": _decode_steps_detail(
            route_plans=route_plans,
            references=references,
            replay_references=replay_references,
            cache_update_indices=cache_update_indices,
            attention_window_indices=attention_window_indices,
            rope_position_indices=rope_position_indices,
            seq_len=seq_len,
            attention_mode=attention_mode,
            cache_update_api=cache_update_api,
            rope_position_api=rope_position_api,
        ),
        "selected_source_keys": [item.source_key for item in metadata],
        "loaded_tensors": [_metadata_summary(item) for item in metadata],
        "loaded_tensor_groups": loaded_groups,
        "loaded_real_tensor_groups": loaded_groups,
        "payload_bytes": {
            "attention_query": loaded_groups["attention_query"]["payload_bytes"],
            "attention_output": loaded_groups["attention_output"]["payload_bytes"],
            "kv_projection": loaded_groups["kv_projection"]["payload_bytes"],
            "ffn_norm": loaded_groups["ffn_norm"]["payload_bytes"],
            "router_selector": loaded_groups["router_selector"]["payload_bytes"],
            "shared_expert": loaded_groups["shared_expert"]["payload_bytes"],
            "routed_experts": loaded_groups["routed_experts"]["payload_bytes"],
            "total": sum(item.nbytes for item in metadata),
        },
        "decoded_tensors": {
            "attn_norm": _tensor_summary(weights.attn_norm),
            "q_norm": _tensor_summary(weights.attention.q_norm),
            "wq_a": _tensor_summary(weights.attention.wq_a),
            "wq_b": _tensor_summary(weights.attention.wq_b),
            "wo_a": _tensor_summary(weights.attention.wo_a),
            "wo_b": _tensor_summary(weights.attention.wo_b),
            "wkv": _tensor_summary(weights.kv.wkv),
            "kv_norm": _tensor_summary(weights.kv.kv_norm),
            "kv_cache_initial": _tensor_summary(kv_cache_initial),
            "ffn_norm": _tensor_summary(weights.ffn_norm),
            "router_gate": _tensor_summary(weights.router_gate),
            "router_bias": _optional_tensor_summary(weights.router_bias),
            "router_tid2eid": _optional_tensor_summary(weights.router_tid2eid),
            "shared_w1": _tensor_summary(weights.shared_expert["w1"]),
            "shared_w2": _tensor_summary(weights.shared_expert["w2"]),
            "shared_w3": _tensor_summary(weights.shared_expert["w3"]),
            "routed_experts": {
                str(expert): {projection: _tensor_summary(weight) for projection, weight in expert_weights.items()}
                for expert, expert_weights in weights.routed_experts.items()
            },
        },
        "budget": {
            "max_tensors": int(max_tensors),
            "max_bytes": int(max_bytes),
            "selected_tensors": len(metadata),
            "selected_payload_bytes": sum(item.nbytes for item in metadata),
        },
        "trace_capture": {
            "attempted": False,
            "capture_passed": False,
            "execute_replay_attempted": False,
            "execute_replay_passed": False,
            "trace_region_size": int(trace_region_size),
            "guard_enabled": True,
            "guarded_symbols": guarded_labels,
            "ttnn_to_torch_guarded": "ttnn.to_torch" in guarded_labels,
            "host_boundaries_inside_trace": [],
            "traced_operations": [],
            "decode_step_count": int(decode_steps),
            "capture_count": 0,
            "positions_used": [int(value) for value in cache_update_indices],
            "one_trace_capture_replayed_across_positions": False,
            "single_capture_replayed_across_positions": False,
            "recaptured_per_position": False,
            "cache_update_api": cache_update_api,
            "rope_position_api": rope_position_api,
            "update_index_source": update_index_source,
            "cache_update_index_dynamic": uses_device_update_index,
            "cache_update_index_kind": update_index_kind,
            "cache_read_window_dynamic": False,
            "cache_read_window_status": cache_read_window_status,
            "rope_position_dynamic": uses_device_rope_position,
            "rope_position_kind": rope_position_kind,
            "rope_position_status": rope_position_status,
            "rope_positions_used": [int(value) for value in rope_position_indices],
        },
        "trace_capture_attempted": False,
        "trace_capture_passed": False,
        "trace_execute_replay_passed": False,
        "guard_status": _guard_status(
            {
                "guard_enabled": True,
                "guarded_symbols": guarded_labels,
                "ttnn_to_torch_guarded": "ttnn.to_torch" in guarded_labels,
                "host_boundaries_inside_trace": [],
            }
        ),
        "host_boundaries": [
            {
                "name": "real_weight_decode_to_bf16",
                "location": "before protected traceable decode region",
                "description": "FP8 attention/KV/shared-expert weights and scales are decoded on host before TTNN module setup",
            },
            {
                "name": "routed_fp4_decode_to_bf16",
                "location": "before protected traceable decode region",
                "description": "preselected routed expert FP4 weights and scales are decoded on host before TTNN module setup",
            },
            {
                "name": "router_static_dispatch_preflight",
                "location": "before protected traceable decode region",
                "description": (
                    "router scoring/top-k runs on host before trace only to choose the static expert modules to load "
                    "and to provide torch validation targets; the protected trace recomputes supported router "
                    "logits, scores, top-k/indices, and route weights on device"
                ),
            },
            {
                "name": "router_decode_row_mask_host_to_device",
                "location": "before trace capture",
                "description": (
                    "a fixed row mask is uploaded during module setup so device-computed route weights only affect "
                    "the logical decode token inside the static trace shape"
                ),
            },
            {
                "name": "kv_cache_seed_host_to_device",
                "location": "before trace capture",
                "description": (
                    "a deterministic compressed K/V cache seed is uploaded during module setup; the protected trace "
                    "updates and reads a fixed cache window from this device-resident cache"
                ),
            },
            {
                "name": "rope_table_host_to_device",
                "location": "before trace capture",
                "description": (
                    "DeepSeek V4 RoPE cos/sin data and the rotation matrix are materialized on host and uploaded "
                    "during module setup; Q/K split and rotation execute inside the protected trace"
                ),
            },
            *(
                [
                    {
                        "name": "rope_position_index_host_to_device",
                        "location": "before trace capture and before each replay",
                        "description": (
                            "rope_position_idxs_tensor contents are copied from host into a preallocated device "
                            "tensor outside the guard; the captured ttnn.embedding ops gather cos/sin rows from "
                            "that tensor during replay"
                        ),
                    }
                ]
                if uses_device_rope_position
                else []
            ),
            {
                "name": "activation_host_to_device",
                "location": "before trace capture and before replay",
                "description": "static-shape decode activation is copied into a preallocated device tensor outside the guard",
            },
            *(
                [
                    {
                        "name": "cache_update_index_host_to_device",
                        "location": "before trace capture and before each replay",
                        "description": (
                            "update_idxs_tensor contents are copied from host into a preallocated device tensor "
                            "outside the guard; the captured paged_update_cache op reads that tensor during replay"
                        ),
                    }
                ]
                if uses_device_update_index
                else [
                    {
                        "name": "trace_recapture_per_position",
                        "location": "outside protected trace body",
                        "description": (
                            "multi-position smoke captures the same guarded TTNN body once per static cache index "
                            "because the scalar update_cache path does not expose the cache write index as a "
                            "replay-mutable device input"
                        ),
                    }
                ]
            ),
            {
                "name": "trace_output_readback",
                "location": "after trace replay",
                "description": "TTNN outputs are copied to host after replay for accuracy checks only",
            },
        ],
        "host_boundaries_inside_trace": [],
        "host_boundaries_outside_trace": [
            "real_weight_decode_to_bf16",
            "routed_fp4_decode_to_bf16",
            "router_static_dispatch_preflight",
            "router_decode_row_mask_host_to_device",
            "kv_cache_seed_host_to_device",
            "rope_table_host_to_device",
            *(["rope_position_index_host_to_device"] if uses_device_rope_position else []),
            "activation_host_to_device",
            "cache_update_index_host_to_device" if uses_device_update_index else "trace_recapture_per_position",
            "trace_output_readback",
        ],
        "reference_ops": _traceable_decode_reference_ops(
            attention_mode,
            config=config,
            weights=weights,
            route_plan=route_plan,
        ),
        "ttnn_ops": [],
        "inputs": {
            "capture_activation": _tensor_summary(activation),
            "kv_cache_initial": _tensor_summary(kv_cache_initial),
            "replay_activation": _tensor_summary(replay_activation),
            "capture_activations_by_step": [_tensor_summary(value) for value in activations],
            "replay_activations_by_step": [_tensor_summary(value) for value in replay_activations],
        },
        "reference": {name: _tensor_summary(value) for name, value in reference.items()},
        "replay_reference": {name: _tensor_summary(value) for name, value in replay_reference.items()},
        "reference_by_step": [
            {
                "step": step,
                "position": int(cache_update_indices[step]),
                "tensors": {name: _tensor_summary(value) for name, value in step_reference.items()},
            }
            for step, step_reference in enumerate(references)
        ],
        "replay_reference_by_step": [
            {
                "step": step,
                "position": int(cache_update_indices[step]),
                "tensors": {name: _tensor_summary(value) for name, value in step_reference.items()},
            }
            for step, step_reference in enumerate(replay_references)
        ],
        "ttnn": {},
        "ttnn_by_step": [],
        "accuracy": {},
        "accuracy_by_step": [],
        "passed": False,
    }


def _metadata_groups(
    metadata: Sequence[TensorMetadata],
    keys: Mapping[str, Sequence[str]],
) -> dict[str, list[TensorMetadata]]:
    groups = {name: [] for name in keys}
    key_to_group = {key: name for name, group_keys in keys.items() for key in group_keys}
    for item in metadata:
        group = key_to_group.get(item.canonical_key)
        if group is None:
            raise ValueError(f"Unexpected tensor in traceable decode slice: {item.canonical_key}")
        groups[group].append(item)
    return groups


def _decode_steps_detail(
    *,
    route_plans: Sequence[TraceableDecodeRoutePlan],
    references: Sequence[Mapping[str, torch.Tensor]],
    replay_references: Sequence[Mapping[str, torch.Tensor]],
    cache_update_indices: Sequence[int],
    attention_window_indices: Sequence[int],
    rope_position_indices: Sequence[int],
    seq_len: int,
    attention_mode: str,
    cache_update_api: str,
    rope_position_api: str,
) -> list[dict[str, Any]]:
    cache_update_api = _validate_cache_update_api(cache_update_api)
    rope_position_api = _validate_rope_position_api(rope_position_api)
    update_index_kind = (
        "replay_mutable_device_tensor"
        if cache_update_api == TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR
        else "static_host_argument_per_trace_capture"
    )
    rope_position_kind = (
        "replay_mutable_device_tensor"
        if rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR
        else "static_host_materialized_tables"
    )
    details = []
    for step, (
        route_plan,
        reference,
        replay_reference,
        cache_update_index,
        attention_window_index,
        rope_position_index,
    ) in enumerate(
        zip(
            route_plans,
            references,
            replay_references,
            cache_update_indices,
            attention_window_indices,
            rope_position_indices,
        )
    ):
        position = int(cache_update_index)
        window_start = int(attention_window_index)
        rope_position = int(rope_position_index)
        details.append(
            {
                "step": step,
                "position": position,
                "cache_update_index": position,
                "cache_update_api": cache_update_api,
                "rope_position_api": rope_position_api,
                "update_index_source": "device_tensor"
                if cache_update_api == TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR
                else "host_scalar",
                "cache_update_index_kind": update_index_kind,
                "cache_rows_updated": [position],
                "cache_window_rows": [window_start, window_start + int(seq_len)],
                "cache_read_window_dynamic": False,
                "rope_position_index": rope_position,
                "rope_position_rows": [rope_position, rope_position + int(seq_len)],
                "rope_position_dynamic": rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR,
                "rope_position_kind": rope_position_kind,
                "single_capture_replayed_across_this_position": False,
                "carried_device_kv_cache_state": True,
                "attention_path_stayed_in_trace": True,
                "router_weights_stayed_in_trace": True,
                "router_expert_dispatch_dynamic_in_trace": False,
                "preflight_topk_expert_ids": [int(value) for value in route_plan.router_indices.reshape(-1).tolist()],
                "preflight_topk_route_weights": [
                    float(value) for value in route_plan.router_weights.reshape(-1).float().tolist()
                ],
                "selected_expert_ids": [int(value) for value in route_plan.selected_indices.reshape(-1).tolist()],
                "selected_route_weights": [
                    float(value) for value in route_plan.selected_weights.reshape(-1).float().tolist()
                ],
                "replay_topk_expert_ids": [
                    int(value) for value in replay_reference["router_decode_topk_indices"].reshape(-1).tolist()
                ],
                "replay_topk_route_weights": [
                    float(value)
                    for value in replay_reference["router_decode_route_weights"].reshape(-1).float().tolist()
                ],
                "reference_cache_row_updated": _tensor_summary(reference["kv_cache"][:, :, position : position + 1, :]),
                "replay_reference_cache_row_updated": _tensor_summary(
                    replay_reference["kv_cache"][:, :, position : position + 1, :]
                ),
                "accuracy_focus_keys": [
                    "kv_output",
                    "kv_cache",
                    "attention_cache_window",
                    "rope_cos",
                    "rope_sin",
                    "attention_context_heads"
                    if attention_mode == TRACEABLE_DECODE_ATTENTION_QK_SOFTMAX_MODE
                    else "expanded_attention_cache",
                    "attention_output",
                    "attention_projected",
                    "post_attention_residual",
                    "router_decode_topk_indices",
                    "router_decode_route_weights",
                    "shared_output",
                    "routed_output",
                    "combined_ffn_output",
                    "residual_output",
                ],
                "accuracy": {},
            }
        )
    return details


def _unique_ints(values) -> list[int]:
    seen: set[int] = set()
    unique: list[int] = []
    for value in values:
        int_value = int(value)
        if int_value not in seen:
            seen.add(int_value)
            unique.append(int_value)
    return unique


def _resolve_patch_target(symbol: GuardedSymbol) -> tuple[object, str]:
    target = importlib.import_module(symbol.module_name)
    path_parts = symbol.attr_path.split(".")
    for part in path_parts[:-1]:
        target = getattr(target, part)
    return target, path_parts[-1]


def _blocked_host_boundary(label: str):
    def blocked(*args, **kwargs):
        raise TraceableDecodeHostFallbackError(
            f"Host readback/fallback helper {label!r} is not allowed inside the traceable decode protected region"
        )

    return blocked


def _to_tt_norm_weight(
    weight: torch.Tensor,
    *,
    device,
    dtype,
    memory_config,
):
    return ttnn.from_torch(
        weight.contiguous().to(torch.bfloat16),
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )


def _to_tt_linear_weight(
    weight: torch.Tensor,
    *,
    device,
    dtype,
    memory_config,
):
    torch_weight = weight.transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    return ttnn.from_torch(
        torch_weight,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )


def _to_tt_kv_cache(
    *,
    cache_len: int,
    kv_output_dim: int,
    initial_cache: torch.Tensor | None = None,
    device,
    dtype,
    memory_config,
):
    if initial_cache is None:
        cache = torch.zeros((1, 1, int(cache_len), int(kv_output_dim)), dtype=torch.bfloat16)
    else:
        _validate_kv_cache_initial(initial_cache, cache_len=cache_len, kv_output_dim=kv_output_dim)
        cache = initial_cache.contiguous().to(torch.bfloat16)
    return ttnn.from_torch(
        cache,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )


def _to_tt_router_bias(
    bias: torch.Tensor,
    *,
    token_rows: int,
    device,
    dtype,
    memory_config,
):
    if bias.ndim != 1:
        raise ValueError(f"router bias must have shape [experts], got {tuple(bias.shape)}")
    expanded = bias.reshape(1, 1, 1, -1).expand(1, 1, int(token_rows), int(bias.shape[0])).contiguous()
    return ttnn.from_torch(
        expanded.to(torch.bfloat16),
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )


def _to_tt_static_router_indices(
    indices: torch.Tensor,
    *,
    token_rows: int,
    device,
    memory_config,
):
    if indices.ndim != 4 or tuple(indices.shape[:3]) != (1, 1, 1):
        raise ValueError(f"router indices must have shape [1, 1, 1, topk], got {tuple(indices.shape)}")
    if int(indices.max().item()) > 65535:
        raise ValueError("static router indices exceed ttnn.uint16 range")
    expanded = indices.to(torch.uint16).expand(1, 1, int(token_rows), int(indices.shape[-1])).contiguous()
    return ttnn.from_torch(
        expanded,
        device=device,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )


def _to_tt_router_decode_row_mask(
    *,
    token_rows: int,
    decode_token_index: int,
    device,
    dtype,
    memory_config,
):
    if not 0 <= int(decode_token_index) < int(token_rows):
        raise ValueError(f"decode_token_index must be in [0, {token_rows}), got {decode_token_index}")
    mask = torch.zeros((1, 1, int(token_rows), 1), dtype=torch.bfloat16)
    mask[0, 0, int(decode_token_index), 0] = 1
    return ttnn.from_torch(
        mask,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )


def _ttnn_router_path(
    ffn_norm_output,
    *,
    memory_config,
    router_gate,
    router_bias,
    static_router_indices,
    config: DeepSeekV4FlashConfig,
) -> dict[str, object]:
    router_logits = ttnn.linear(ffn_norm_output, router_gate, memory_config=memory_config)
    router_scores = _ttnn_router_scores(
        router_logits,
        scoring_func=str(config.scoring_func),
        memory_config=memory_config,
    )
    router_selection_scores = (
        ttnn.add(router_scores, router_bias, memory_config=memory_config) if router_bias is not None else router_scores
    )
    if static_router_indices is None:
        router_topk_selection_scores, router_topk_indices = ttnn.topk(
            router_selection_scores,
            k=int(config.num_experts_per_tok),
            dim=-1,
            largest=True,
            sorted=True,
            memory_config=memory_config,
        )
    else:
        router_topk_indices = static_router_indices
        router_topk_selection_scores = ttnn.gather(router_selection_scores, dim=3, index=router_topk_indices)

    if router_bias is None and static_router_indices is None:
        router_topk_route_scores = router_topk_selection_scores
    else:
        router_topk_route_scores = ttnn.gather(router_scores, dim=3, index=router_topk_indices)

    if str(config.scoring_func) == "softmax":
        router_route_weights = ttnn.mul(
            router_topk_route_scores,
            float(config.routed_scaling_factor),
            memory_config=memory_config,
        )
    else:
        route_weight_sum = ttnn.sum(router_topk_route_scores, dim=3, keepdim=True) + 1e-20
        router_route_weights = ttnn.div(router_topk_route_scores, route_weight_sum, memory_config=memory_config)
        router_route_weights = ttnn.mul(
            router_route_weights,
            float(config.routed_scaling_factor),
            memory_config=memory_config,
        )
    return {
        "router_logits": router_logits,
        "router_scores": router_scores,
        "router_selection_scores": router_selection_scores,
        "router_topk_selection_scores": router_topk_selection_scores,
        "router_topk_indices": router_topk_indices,
        "router_topk_route_scores": router_topk_route_scores,
        "router_route_weights": router_route_weights,
    }


def _ttnn_router_scores(router_logits, *, scoring_func: str, memory_config):
    if scoring_func == "softmax":
        return ttnn.softmax(router_logits, dim=-1, memory_config=memory_config)
    if scoring_func == "sigmoid":
        return ttnn.sigmoid(router_logits, memory_config=memory_config)
    if scoring_func == "sqrtsoftplus":
        return ttnn.sqrt(ttnn.softplus(router_logits, memory_config=memory_config), memory_config=memory_config)
    raise ValueError(f"Unsupported DeepSeek V4 Flash scoring_func {scoring_func!r}")


def _ttnn_route_weight_for_topk_slot(
    route_weights,
    *,
    slot: int,
    route_mask,
    intermediate_size: int,
    memory_config,
):
    if slot < 0 or slot >= int(route_weights.shape[-1]):
        raise ValueError(f"route slot {slot} is outside route_weights top-k width {route_weights.shape[-1]}")
    token_rows = int(route_weights.shape[-2])
    slot_weight = ttnn.slice(
        route_weights,
        (0, 0, 0, int(slot)),
        (1, 1, token_rows, int(slot) + 1),
        memory_config=memory_config,
    )
    slot_weight = ttnn.mul(slot_weight, route_mask, memory_config=memory_config)
    return ttnn.repeat(slot_weight, ttnn.Shape((1, 1, 1, int(intermediate_size))))


def _ttnn_topk_prefix(route_weights, *, topk_prefix: int, memory_config):
    if int(topk_prefix) == int(route_weights.shape[-1]):
        return route_weights
    if int(topk_prefix) <= 0 or int(topk_prefix) > int(route_weights.shape[-1]):
        raise ValueError(f"topk_prefix must be in [1, {route_weights.shape[-1]}], got {topk_prefix}")
    return ttnn.slice(
        route_weights,
        (0, 0, 0, 0),
        (1, 1, int(route_weights.shape[-2]), int(topk_prefix)),
        memory_config=memory_config,
    )


def _ttnn_decode_token_slice(tensor, *, decode_token_index: int, memory_config):
    if not 0 <= int(decode_token_index) < int(tensor.shape[-2]):
        raise ValueError(f"decode_token_index must be in [0, {tensor.shape[-2]}), got {decode_token_index}")
    return ttnn.slice(
        tensor,
        (0, 0, int(decode_token_index), 0),
        (1, 1, int(decode_token_index) + 1, int(tensor.shape[-1])),
        memory_config=memory_config,
    )


def _to_tt_rope_tensors(
    config: DeepSeekV4FlashConfig,
    *,
    layer: int,
    start_pos: int,
    seq_len: int,
    device,
    dtype,
    memory_config,
):
    freqs = precompute_deepseek_v4_rope_frequencies(
        config,
        layer=layer,
        seq_len=int(start_pos) + int(seq_len),
    )[int(start_pos) : int(start_pos) + int(seq_len)]
    cos = torch.stack((freqs.real, freqs.real), dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack((freqs.imag, freqs.imag), dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    trans_mat = torch.zeros(1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE, dtype=torch.bfloat16)
    trans_mat[..., torch.arange(0, ttnn.TILE_SIZE, 2), torch.arange(1, ttnn.TILE_SIZE, 2)] = 1
    trans_mat[..., torch.arange(1, ttnn.TILE_SIZE, 2), torch.arange(0, ttnn.TILE_SIZE, 2)] = -1
    return (
        ttnn.from_torch(
            cos.contiguous().to(torch.bfloat16),
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        ),
        ttnn.from_torch(
            sin.contiguous().to(torch.bfloat16),
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        ),
        ttnn.from_torch(
            trans_mat,
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        ),
    )


def _torch_rope_cos_sin(
    config: DeepSeekV4FlashConfig,
    *,
    layer: int,
    start_pos: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = precompute_deepseek_v4_rope_frequencies(
        config,
        layer=layer,
        seq_len=int(start_pos) + int(seq_len),
    )[int(start_pos) : int(start_pos) + int(seq_len)]
    cos = torch.stack((freqs.real, freqs.real), dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack((freqs.imag, freqs.imag), dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos.contiguous().to(torch.bfloat16), sin.contiguous().to(torch.bfloat16)


def _to_tt_rope_embedding_tables(
    config: DeepSeekV4FlashConfig,
    *,
    layer: int,
    max_position: int,
    device,
    dtype,
    memory_config,
):
    freqs = precompute_deepseek_v4_rope_frequencies(config, layer=layer, seq_len=int(max_position))
    cos = torch.stack((freqs.real, freqs.real), dim=-1).flatten(-2)
    sin = torch.stack((freqs.imag, freqs.imag), dim=-1).flatten(-2)
    return (
        ttnn.from_torch(
            cos.contiguous().to(torch.bfloat16),
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        ),
        ttnn.from_torch(
            sin.contiguous().to(torch.bfloat16),
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        ),
        ttnn.from_torch(
            _rope_transformation_matrix(),
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
        ),
    )


def _ttnn_gather_rope_tensors(position_idxs, cos_table, sin_table, *, memory_config):
    cos = ttnn.embedding(position_idxs, cos_table, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)
    sin = ttnn.embedding(position_idxs, sin_table, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)
    return ttnn.unsqueeze_to_4D(cos), ttnn.unsqueeze_to_4D(sin)


def _rope_transformation_matrix() -> torch.Tensor:
    trans_mat = torch.zeros(1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE, dtype=torch.bfloat16)
    trans_mat[..., torch.arange(0, ttnn.TILE_SIZE, 2), torch.arange(1, ttnn.TILE_SIZE, 2)] = 1
    trans_mat[..., torch.arange(1, ttnn.TILE_SIZE, 2), torch.arange(0, ttnn.TILE_SIZE, 2)] = -1
    return trans_mat


def _sum_ttnn_tensors(tensors, *, memory_config):
    values = list(tensors)
    if not values:
        raise ValueError("at least one routed expert output is required")
    result = values[0]
    for value in values[1:]:
        result = ttnn.add(result, value, memory_config=memory_config)
    return result


def _ttnn_fixed_window_attention(
    *,
    q_output,
    attention_cache_window,
    attention_mode: str,
    config: DeepSeekV4FlashConfig,
    q_head_norm,
    rope_cos,
    rope_sin,
    rope_trans_mat,
    memory_config,
) -> dict[str, object]:
    attention_mode = _validate_attention_mode(attention_mode)
    if attention_mode == TRACEABLE_DECODE_ATTENTION_LEGACY_BLEND_MODE:
        repeat_factor = _attention_cache_repeat_factor(config)
        if repeat_factor == 1:
            expanded_attention_cache = attention_cache_window
        else:
            expanded_attention_cache = ttnn.repeat(
                attention_cache_window,
                ttnn.Shape([1, 1, 1, repeat_factor]),
            )
        return {
            "expanded_attention_cache": expanded_attention_cache,
            "attention_output": ttnn.add(q_output, expanded_attention_cache, memory_config=memory_config),
        }

    _validate_qk_softmax_attention_config(config)
    token_count = int(q_output.shape[-2])
    num_heads = int(config.num_attention_heads)
    head_dim = int(config.head_dim)
    rope_dim = int(config.qk_rope_head_dim)
    nope_dim = head_dim - rope_dim
    attention_output_dim = num_heads * head_dim
    if int(q_output.shape[-1]) != attention_output_dim:
        raise ValueError(f"q_output width must be {attention_output_dim}, got {q_output.shape[-1]}")
    if int(attention_cache_window.shape[-1]) != head_dim:
        raise ValueError(f"attention cache window width must be {head_dim}, got {attention_cache_window.shape[-1]}")

    q_heads_token_major = ttnn.reshape(q_output, (1, token_count, num_heads, head_dim))
    q_heads_pre_norm = ttnn.transpose(q_heads_token_major, 1, 2, memory_config=memory_config)
    q_heads_norm = ttnn.rms_norm(
        q_heads_pre_norm,
        weight=q_head_norm,
        epsilon=float(config.rms_norm_eps),
        memory_config=memory_config,
    )
    q_nope = ttnn.slice(
        q_heads_norm,
        (0, 0, 0, 0),
        (1, num_heads, token_count, nope_dim),
        memory_config=memory_config,
    )
    q_rope = ttnn.slice(
        q_heads_norm,
        (0, 0, 0, nope_dim),
        (1, num_heads, token_count, head_dim),
        memory_config=memory_config,
    )
    q_rope_rotated = ttnn.experimental.rotary_embedding_llama(
        q_rope,
        rope_cos,
        rope_sin,
        rope_trans_mat,
        is_decode_mode=False,
        memory_config=memory_config,
    )
    q_heads = ttnn.concat([q_nope, q_rope_rotated], dim=-1, memory_config=memory_config)

    key_cache_nope = ttnn.slice(
        attention_cache_window,
        (0, 0, 0, 0),
        (1, 1, token_count, nope_dim),
        memory_config=memory_config,
    )
    key_cache_rope = ttnn.slice(
        attention_cache_window,
        (0, 0, 0, nope_dim),
        (1, 1, token_count, head_dim),
        memory_config=memory_config,
    )
    key_cache_rope_rotated = ttnn.experimental.rotary_embedding_llama(
        key_cache_rope,
        rope_cos,
        rope_sin,
        rope_trans_mat,
        is_decode_mode=False,
        memory_config=memory_config,
    )
    key_cache = ttnn.concat([key_cache_nope, key_cache_rope_rotated], dim=-1, memory_config=memory_config)
    key_heads = ttnn.repeat(key_cache, ttnn.Shape([1, num_heads, 1, 1]))
    value_heads = ttnn.repeat(attention_cache_window, ttnn.Shape([1, num_heads, 1, 1]))
    key_heads_transposed = ttnn.transpose(key_heads, -2, -1, memory_config=memory_config)
    qk_scores = ttnn.matmul(q_heads, key_heads_transposed, memory_config=memory_config)
    qk_scores = ttnn.mul(qk_scores, 1.0 / math.sqrt(float(head_dim)), memory_config=memory_config)
    attention_probs = ttnn.softmax(qk_scores, dim=-1, memory_config=memory_config)
    attention_context_heads = ttnn.matmul(attention_probs, value_heads, memory_config=memory_config)
    attention_context_token_major = ttnn.transpose(attention_context_heads, 1, 2, memory_config=memory_config)
    attention_output = ttnn.reshape(attention_context_token_major, (1, 1, token_count, attention_output_dim))
    return {
        "attention_q_heads_pre_norm": q_heads_pre_norm,
        "attention_q_heads_norm": q_heads_norm,
        "attention_q_nope": q_nope,
        "attention_q_rope": q_rope,
        "attention_q_rope_rotated": q_rope_rotated,
        "attention_q_heads": q_heads,
        "attention_key_cache_nope": key_cache_nope,
        "attention_key_cache_rope": key_cache_rope,
        "attention_key_cache_rope_rotated": key_cache_rope_rotated,
        "attention_key_cache": key_cache,
        "attention_key_heads": key_heads,
        "attention_value_heads": value_heads,
        "qk_scores": qk_scores,
        "attention_probs": attention_probs,
        "attention_context_heads": attention_context_heads,
        "attention_output": attention_output,
    }


def _torch_fixed_window_attention(
    *,
    q_output: torch.Tensor,
    attention_cache_window: torch.Tensor,
    attention_mode: str,
    config: DeepSeekV4FlashConfig,
    layer: int,
    start_pos: int,
) -> dict[str, torch.Tensor]:
    attention_mode = _validate_attention_mode(attention_mode)
    if attention_mode == TRACEABLE_DECODE_ATTENTION_LEGACY_BLEND_MODE:
        expanded_attention_cache = attention_cache_window.repeat(1, 1, 1, _attention_cache_repeat_factor(config))
        return {
            "expanded_attention_cache": expanded_attention_cache,
            "attention_output": (q_output.float() + expanded_attention_cache.float()).to(torch.bfloat16),
        }

    _validate_qk_softmax_attention_config(config)
    batch_size, _, token_count, attention_output_dim = q_output.shape
    num_heads = int(config.num_attention_heads)
    head_dim = int(config.head_dim)
    rope_dim = int(config.qk_rope_head_dim)
    nope_dim = head_dim - rope_dim
    expected_output_dim = num_heads * head_dim
    if int(attention_output_dim) != expected_output_dim:
        raise ValueError(f"q_output width must be {expected_output_dim}, got {attention_output_dim}")
    if int(attention_cache_window.shape[-1]) != head_dim:
        raise ValueError(f"attention cache window width must be {head_dim}, got {attention_cache_window.shape[-1]}")

    freqs_cis = precompute_deepseek_v4_rope_frequencies(
        config,
        layer=layer,
        seq_len=int(start_pos) + int(token_count),
    )[int(start_pos) : int(start_pos) + int(token_count)]
    q_heads_token_major_pre_norm = q_output[:, 0].reshape(batch_size, token_count, num_heads, head_dim).contiguous()
    q_heads_token_major = rms_norm(
        q_heads_token_major_pre_norm,
        torch.ones(head_dim, dtype=torch.bfloat16),
        eps=float(config.rms_norm_eps),
    )
    q_nope_token_major, q_rope_token_major = q_heads_token_major.split([nope_dim, rope_dim], dim=-1)
    q_rope_rotated_token_major = apply_deepseek_v4_rotary(q_rope_token_major.contiguous(), freqs_cis)
    q_heads = torch.cat([q_nope_token_major, q_rope_rotated_token_major], dim=-1).transpose(1, 2).contiguous()

    key_cache_projection = attention_cache_window[:, 0].contiguous()
    key_cache_nope, key_cache_rope = key_cache_projection.split([nope_dim, rope_dim], dim=-1)
    key_cache_rope_rotated = apply_deepseek_v4_rotary(key_cache_rope.contiguous(), freqs_cis)
    key_cache = torch.cat([key_cache_nope, key_cache_rope_rotated], dim=-1).contiguous()
    key_heads = key_cache.unsqueeze(1).repeat(1, num_heads, 1, 1).contiguous()
    value_heads = attention_cache_window.repeat(1, num_heads, 1, 1).contiguous()
    qk_scores = torch.matmul(q_heads.float(), key_heads.transpose(-2, -1).float())
    qk_scores = (qk_scores * (1.0 / math.sqrt(float(head_dim)))).to(torch.bfloat16)
    attention_probs = torch.softmax(qk_scores.float(), dim=-1).to(torch.bfloat16)
    attention_context_heads = torch.matmul(attention_probs.float(), value_heads.float()).to(torch.bfloat16)
    attention_output = (
        attention_context_heads.transpose(1, 2)
        .reshape(batch_size, 1, token_count, attention_output_dim)
        .to(torch.bfloat16)
    )
    return {
        "attention_q_heads_pre_norm": q_heads_token_major_pre_norm.transpose(1, 2).contiguous().to(torch.bfloat16),
        "attention_q_heads_norm": q_heads_token_major.transpose(1, 2).contiguous().to(torch.bfloat16),
        "attention_q_nope": q_nope_token_major.transpose(1, 2).contiguous().to(torch.bfloat16),
        "attention_q_rope": q_rope_token_major.transpose(1, 2).contiguous().to(torch.bfloat16),
        "attention_q_rope_rotated": q_rope_rotated_token_major.transpose(1, 2).contiguous().to(torch.bfloat16),
        "attention_q_heads": q_heads.to(torch.bfloat16),
        "attention_key_cache_nope": key_cache_nope.unsqueeze(1).contiguous().to(torch.bfloat16),
        "attention_key_cache_rope": key_cache_rope.unsqueeze(1).contiguous().to(torch.bfloat16),
        "attention_key_cache_rope_rotated": key_cache_rope_rotated.unsqueeze(1).contiguous().to(torch.bfloat16),
        "attention_key_cache": key_cache.unsqueeze(1).contiguous().to(torch.bfloat16),
        "attention_key_heads": key_heads.to(torch.bfloat16),
        "attention_value_heads": value_heads.to(torch.bfloat16),
        "qk_scores": qk_scores,
        "attention_probs": attention_probs,
        "attention_context_heads": attention_context_heads,
        "attention_output": attention_output,
    }


def _kv_update_memory_config(*, device, token_rows: int, width: int):
    if token_rows <= 0:
        raise ValueError(f"token_rows must be positive, got {token_rows}")
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    grid_size = device.compute_with_storage_grid_size()
    shard_grid = ttnn.num_cores_to_corerangeset(1, grid_size, row_wise=True)
    return ttnn.create_sharded_memory_config(
        shape=(int(token_rows), int(width)),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _single_core_height_sharded_memory_config(tensor, *, device):
    padded_width = int(tensor.padded_shape[-1])
    if padded_width <= 0:
        raise ValueError(f"tensor padded width must be positive, got {padded_width}")
    shard_height = int(tensor.volume()) // padded_width
    if shard_height <= 0:
        raise ValueError(f"tensor shard height must be positive, got {shard_height}")
    grid_size = device.compute_with_storage_grid_size()
    shard_grid = ttnn.num_cores_to_corerangeset(1, grid_size, row_wise=True)
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        [shard_height, padded_width],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _kv_output_dim(config: DeepSeekV4FlashConfig) -> int:
    return int(config.num_key_value_heads) * int(config.head_dim)


def _attention_cache_repeat_factor(config: DeepSeekV4FlashConfig) -> int:
    q_output_dim = int(config.num_attention_heads) * int(config.head_dim)
    kv_output_dim = _kv_output_dim(config)
    if q_output_dim % kv_output_dim != 0:
        raise ValueError(f"q_output_dim {q_output_dim} must be divisible by kv_output_dim {kv_output_dim}")
    return q_output_dim // kv_output_dim


def _attention_head_repeat_factor(config: DeepSeekV4FlashConfig) -> int:
    if int(config.num_attention_heads) % int(config.num_key_value_heads) != 0:
        raise ValueError(
            "num_attention_heads must be divisible by num_key_value_heads, "
            f"got {config.num_attention_heads} and {config.num_key_value_heads}"
        )
    return int(config.num_attention_heads) // int(config.num_key_value_heads)


def _validate_attention_mode(attention_mode: str) -> str:
    if attention_mode not in TRACEABLE_DECODE_ATTENTION_MODES:
        raise ValueError(f"attention_mode must be one of {TRACEABLE_DECODE_ATTENTION_MODES}, got {attention_mode!r}")
    return str(attention_mode)


def _validate_cache_update_api(cache_update_api: str) -> str:
    if cache_update_api not in TRACEABLE_DECODE_CACHE_UPDATE_APIS:
        raise ValueError(
            f"cache_update_api must be one of {TRACEABLE_DECODE_CACHE_UPDATE_APIS}, got {cache_update_api!r}"
        )
    return str(cache_update_api)


def _validate_rope_position_api(rope_position_api: str) -> str:
    if rope_position_api not in TRACEABLE_DECODE_ROPE_POSITION_APIS:
        raise ValueError(
            f"rope_position_api must be one of {TRACEABLE_DECODE_ROPE_POSITION_APIS}, got {rope_position_api!r}"
        )
    return str(rope_position_api)


def _validate_qk_softmax_attention_config(config: DeepSeekV4FlashConfig) -> None:
    if int(config.num_key_value_heads) != 1:
        raise ValueError(
            "fixed-window QK softmax trace currently expects one compressed KV head, "
            f"got num_key_value_heads={config.num_key_value_heads}; use "
            f"{TRACEABLE_DECODE_ATTENTION_LEGACY_BLEND_MODE!r} for the legacy stepping stone"
        )
    if _kv_output_dim(config) != int(config.head_dim):
        raise ValueError(
            f"fixed-window QK softmax trace expects kv_output_dim=head_dim={config.head_dim}, "
            f"got kv_output_dim={_kv_output_dim(config)}"
        )
    if int(config.qk_rope_head_dim) <= 0 or int(config.qk_rope_head_dim) >= int(config.head_dim):
        raise ValueError(
            "fixed-window QK softmax trace expects a strict nope/rope split, "
            f"got qk_rope_head_dim={config.qk_rope_head_dim} and head_dim={config.head_dim}"
        )
    if int(config.qk_rope_head_dim) % 2 != 0:
        raise ValueError(f"qk_rope_head_dim must be even for RoPE, got {config.qk_rope_head_dim}")


def _traceable_decode_common_ops(cache_update_api: str) -> list[str]:
    cache_update_api = _validate_cache_update_api(cache_update_api)
    ops = [
        "ttnn.rms_norm(attn_norm)",
        "TtAttentionProjection.project_q_rank",
        "TtAttentionProjection.project_q_from_rank",
        "ttnn.linear(wkv)",
        "ttnn.rms_norm(kv_norm)",
    ]
    if cache_update_api == TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR:
        ops.extend(
            [
                "ttnn.slice(kv_update_decode_token)",
                "ttnn.to_memory_config(kv_update_decode_token_height_sharded)",
                "ttnn.experimental.paged_update_cache(kv_projection_cache,update_idxs_tensor)",
            ]
        )
    else:
        ops.extend(
            [
                "ttnn.to_memory_config(kv_update_height_sharded)",
                "ttnn.update_cache(kv_projection_cache)",
            ]
        )
    ops.append("ttnn.slice(kv_cache_fixed_window)")
    return ops


def _traceable_decode_attention_ops(
    attention_mode: str,
    *,
    rope_position_api: str = DEFAULT_TRACEABLE_DECODE_ROPE_POSITION_API,
) -> list[str]:
    attention_mode = _validate_attention_mode(attention_mode)
    rope_position_api = _validate_rope_position_api(rope_position_api)
    if attention_mode == TRACEABLE_DECODE_ATTENTION_LEGACY_BLEND_MODE:
        return [
            "ttnn.repeat(kv_cache_window_to_attention_width)",
            "ttnn.add(q_output,expanded_kv_cache_window)",
        ]
    ops = []
    if rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR:
        ops.extend(
            [
                "ttnn.embedding(rope_position_idxs,rope_cos_table)",
                "ttnn.embedding(rope_position_idxs,rope_sin_table)",
                "ttnn.unsqueeze_to_4D(dynamic_rope_cos_sin)",
            ]
        )
    ops.extend(
        [
            "ttnn.reshape(q_output_to_q_heads_token_major)",
            "ttnn.transpose(q_heads_token_major_to_heads)",
            "ttnn.rms_norm(q_heads)",
            "ttnn.slice(q_nope/q_rope)",
            "ttnn.experimental.rotary_embedding_llama(q_rope)",
            "ttnn.concat(q_nope,q_rope_rotated)",
            "ttnn.slice(kv_cache_window_to_k_nope/k_rope)",
            "ttnn.experimental.rotary_embedding_llama(k_rope)",
            "ttnn.concat(k_nope,k_rope_rotated)",
            "ttnn.repeat(k_cache_to_attention_heads)",
            "ttnn.repeat(v_cache_window_to_attention_heads)",
            "ttnn.transpose(k_heads_to_k_heads_transposed)",
            "ttnn.matmul(q_heads,k_heads_transposed)",
            "ttnn.mul(qk_scores,1/sqrt(head_dim))",
            "ttnn.softmax(qk_scores)",
            "ttnn.matmul(attention_probs,value_heads)",
            "ttnn.transpose(context_heads_to_token_major)",
            "ttnn.reshape(context_heads_to_attention_output)",
        ]
    )
    return ops


def _traceable_decode_router_ops(
    *,
    config: DeepSeekV4FlashConfig,
    weights: TraceableDecodeWeights,
    route_plan: TraceableDecodeRoutePlan,
) -> list[str]:
    ops = ["ttnn.linear(router_gate)"]
    if str(config.scoring_func) == "softmax":
        ops.append("ttnn.softmax(router_logits)")
    elif str(config.scoring_func) == "sigmoid":
        ops.append("ttnn.sigmoid(router_logits)")
    elif str(config.scoring_func) == "sqrtsoftplus":
        ops.extend(["ttnn.softplus(router_logits)", "ttnn.sqrt(router_softplus)"])
    else:
        raise ValueError(f"Unsupported DeepSeek V4 Flash scoring_func {config.scoring_func!r}")
    if weights.router_bias is not None:
        ops.append("ttnn.add(router_scores,router_bias)")
    if weights.router_tid2eid is None:
        ops.append("ttnn.topk(router_selection_scores)")
    else:
        ops.append("ttnn.gather(router_selection_scores,static_tid2eid_indices)")
    if weights.router_bias is not None or weights.router_tid2eid is not None:
        ops.append("ttnn.gather(router_scores,router_topk_indices)")
    if str(config.scoring_func) != "softmax":
        ops.extend(["ttnn.sum(router_topk_route_scores)", "ttnn.div(router_topk_route_scores,router_weight_sum)"])
    ops.append("ttnn.mul(router_route_weights,routed_scaling_factor)")
    if int(route_plan.topk_prefix) != int(route_plan.full_topk):
        ops.append("ttnn.slice(router_route_weights_topk_prefix)")
    ops.append("ttnn.mul(router_selected_route_weights,decode_row_mask)")
    return ops


def _traceable_decode_projection_and_ffn_ops(
    *,
    config: DeepSeekV4FlashConfig,
    weights: TraceableDecodeWeights,
    route_plan: TraceableDecodeRoutePlan,
) -> list[str]:
    return [
        "TtAttentionProjection.project_output",
        "ttnn.slice(attention_output_group_0..N)",
        "ttnn.linear(grouped_wo_a_group_0..N)",
        "ttnn.concat(grouped_wo_a_rank)",
        "ttnn.linear(wo_b)",
        "ttnn.add(hidden,attention_projected)",
        "ttnn.rms_norm(ffn_norm)",
        *_traceable_decode_router_ops(config=config, weights=weights, route_plan=route_plan),
        "TtRoutedExpertMLP(selected_topk_prefix)",
        "ttnn.mul(routed_hidden,device_router_route_weight)",
        "ttnn.add(routed_expert_outputs)",
        "TtSharedExpertMLP",
        "ttnn.add(shared_output,routed_output)",
        "ttnn.add(post_attention_residual,combined_ffn_output)",
    ]


def _traceable_decode_inside_trace_ops(
    attention_mode: str,
    *,
    cache_update_api: str,
    config: DeepSeekV4FlashConfig,
    weights: TraceableDecodeWeights,
    route_plan: TraceableDecodeRoutePlan,
    rope_position_api: str = DEFAULT_TRACEABLE_DECODE_ROPE_POSITION_API,
) -> list[str]:
    return [
        *_traceable_decode_common_ops(cache_update_api),
        *_traceable_decode_attention_ops(attention_mode, rope_position_api=rope_position_api),
        *_traceable_decode_projection_and_ffn_ops(config=config, weights=weights, route_plan=route_plan),
    ]


def _traceable_decode_ttnn_ops(
    attention_mode: str,
    *,
    cache_update_api: str,
    config: DeepSeekV4FlashConfig,
    weights: TraceableDecodeWeights,
    route_plan: TraceableDecodeRoutePlan,
    rope_position_api: str = DEFAULT_TRACEABLE_DECODE_ROPE_POSITION_API,
) -> list[str]:
    return [
        *_traceable_decode_common_ops(cache_update_api),
        *_traceable_decode_attention_ops(attention_mode, rope_position_api=rope_position_api),
        "ttnn.slice(attention_output_group_0..N)",
        "ttnn.linear(grouped_wo_a_group_0..N)",
        "ttnn.concat(grouped_wo_a_rank)",
        "ttnn.linear(wo_b)",
        "ttnn.add(hidden,attention_projected)",
        "ttnn.rms_norm(ffn_norm)",
        *_traceable_decode_router_ops(config=config, weights=weights, route_plan=route_plan),
        "ttnn.linear(shared_w1)",
        "ttnn.linear(shared_w3)",
        "ttnn.mul(silu(shared_gate),shared_up)",
        "ttnn.linear(shared_w2)",
        "ttnn.linear(routed_w1_selected_topk_prefix)",
        "ttnn.linear(routed_w3_selected_topk_prefix)",
        "ttnn.mul(silu(routed_gate),routed_up)",
        "ttnn.mul(routed_hidden,device_router_route_weight)",
        "ttnn.linear(routed_w2_selected_topk_prefix)",
        "ttnn.add(routed_expert_outputs)",
        "ttnn.add(shared_output,routed_output)",
        "ttnn.add(post_attention_residual,combined_ffn_output)",
    ]


def _traceable_decode_reference_ops(
    attention_mode: str,
    *,
    config: DeepSeekV4FlashConfig,
    weights: TraceableDecodeWeights,
    route_plan: TraceableDecodeRoutePlan,
) -> list[str]:
    attention_mode = _validate_attention_mode(attention_mode)
    attention_ops = (
        [
            "torch.fixed_cache_window_repeat",
            "torch.add(q_output,expanded_kv_cache_window)",
        ]
        if attention_mode == TRACEABLE_DECODE_ATTENTION_LEGACY_BLEND_MODE
        else [
            "torch.reshape(q_output_to_q_heads)",
            "torch.rms_norm_reference(q_heads)",
            "torch.split(q_nope/q_rope)",
            "torch.apply_deepseek_v4_rotary(q_rope)",
            "torch.split(k_nope/k_rope)",
            "torch.apply_deepseek_v4_rotary(k_rope)",
            "torch.repeat(k_cache_to_attention_heads)",
            "torch.repeat(v_cache_window_to_attention_heads)",
            "torch.matmul(q_heads,k_heads_transposed)",
            "torch.mul(qk_scores,1/sqrt(head_dim))",
            "torch.softmax(qk_scores)",
            "torch.matmul(attention_probs,value_heads)",
            "torch.reshape(context_heads_to_attention_output)",
        ]
    )
    router_ops = ["torch.linear(router_gate)"]
    if str(config.scoring_func) == "softmax":
        router_ops.append("torch.softmax(router_logits)")
    elif str(config.scoring_func) == "sigmoid":
        router_ops.append("torch.sigmoid(router_logits)")
    elif str(config.scoring_func) == "sqrtsoftplus":
        router_ops.append("torch.sqrt(torch.softplus(router_logits))")
    else:
        raise ValueError(f"Unsupported DeepSeek V4 Flash scoring_func {config.scoring_func!r}")
    if weights.router_bias is not None:
        router_ops.append("torch.add(router_scores,router_bias)")
    router_ops.append(
        "torch.topk(router_selection_scores)" if weights.router_tid2eid is None else "torch.tid2eid_static_indices"
    )
    if weights.router_bias is not None or weights.router_tid2eid is not None:
        router_ops.append("torch.gather(router_scores,router_topk_indices)")
    if str(config.scoring_func) != "softmax":
        router_ops.append("torch.normalize_router_route_weights")
    router_ops.append("torch.mul(router_route_weights,routed_scaling_factor)")
    if int(route_plan.topk_prefix) != int(route_plan.full_topk):
        router_ops.append("torch.slice(router_route_weights_topk_prefix)")
    router_ops.append("torch.mul(router_selected_route_weights,decode_row_mask)")
    return [
        "torch.rms_norm_reference(attn_norm)",
        "torch.linear(wq_a)",
        "torch.rms_norm_reference(q_norm)",
        "torch.linear(wq_b)",
        "torch.linear(wkv)",
        "torch.rms_norm_reference(kv_norm)",
        "torch.cache_update_reference",
        *attention_ops,
        "torch.grouped_output_projection_a",
        "torch.linear(wo_b)",
        "torch.add(hidden,attention_projected)",
        "torch.rms_norm_reference(ffn_norm)",
        *router_ops,
        "torch.routed_swiglu_expert_reference(static_dispatch_device_router_weights)",
        "torch.shared_swiglu_expert_reference",
        "torch.add(shared_output,routed_output)",
        "torch.add(post_attention_residual,combined_ffn_output)",
    ]


def _resolve_cache_update_index(*, seq_len: int, cache_len: int, cache_update_index: int | None) -> int:
    update_index = int(seq_len) if cache_update_index is None else int(cache_update_index)
    if update_index < 0:
        raise ValueError(f"cache_update_index must be non-negative, got {update_index}")
    if update_index >= int(cache_len):
        raise ValueError(f"cache_update_index {update_index} must be less than cache_len {cache_len}")
    return update_index


def _position_dependent_decode_inventory(
    *,
    cache_update_api: str,
    rope_position_api: str,
    config: DeepSeekV4FlashConfig,
) -> dict[str, Any]:
    cache_update_api = _validate_cache_update_api(cache_update_api)
    rope_position_api = _validate_rope_position_api(rope_position_api)
    return {
        "dynamic_cache_write": {
            "status": "used" if cache_update_api == TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR else "available",
            "api": "ttnn.experimental.paged_update_cache(cache, input, update_idxs_tensor=..., page_table=...)",
            "position_input": "mutable device tensor",
            "current_module_path": cache_update_api == TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR,
            "layout_used_here": "[1, 1, 1, kv_output_dim] update, height-sharded on one core",
        },
        "dynamic_rope_position": {
            "status": "used" if rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR else "available",
            "api": (
                "ttnn.embedding(rope_position_idxs, rope_cos_table/sin_table) -> "
                "ttnn.experimental.rotary_embedding_llama"
            ),
            "position_input": "mutable device uint32 tensor [1, seq_len]",
            "current_module_path": rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR,
            "shape_used_here": [
                1,
                "seq_len",
                int(config.qk_rope_head_dim),
            ],
            "next_step": (
                "keep this device position tensor synchronized with the decode current-position tensor used by "
                "paged attention/cache update"
            ),
        },
        "dynamic_cache_read_current_position": {
            "status": "available_not_integrated",
            "api": (
                "ttnn.transformer.paged_scaled_dot_product_attention_decode("
                "q, k_cache, v_cache, page_table_tensor, cur_pos_tensor=...)"
            ),
            "position_input": "mutable device cur_pos_tensor plus device page_table_tensor",
            "current_module_path": False,
            "blocker": (
                "the DeepSeek V4 Flash traceable subpath still materializes a compressed fixed-window cache slice "
                "with ttnn.slice Python bounds and derives K and V from one compressed cache width"
            ),
            "required_shape_step": (
                "feed SDPA decode-style tensors Q [1,b,nh,dh], K/V caches [b,nkv,s,dh] or paged "
                "[blocks,nkv,block,dh], with true separate V channels"
            ),
        },
        "static_bound_ops_remaining": [
            "ttnn.slice(kv_cache_fixed_window) start/end are Python arguments in this module path",
            "static routed expert module dispatch is still selected by host preflight",
            "DeepSeek sparse indexer/sink semantics are not represented by the fixed-window dense attention path",
        ],
    }


def _attention_path_summary(
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    cache_update_index: int,
    seq_len: int,
    attention_mode: str,
    cache_write_index: int | None = None,
    cache_update_api: str = DEFAULT_TRACEABLE_DECODE_CACHE_UPDATE_API,
    rope_position_index: int | None = None,
    rope_position_api: str = DEFAULT_TRACEABLE_DECODE_ROPE_POSITION_API,
) -> dict[str, Any]:
    attention_mode = _validate_attention_mode(attention_mode)
    cache_update_api = _validate_cache_update_api(cache_update_api)
    rope_position_api = _validate_rope_position_api(rope_position_api)
    width_repeat_factor = _attention_cache_repeat_factor(config)
    head_repeat_factor = _attention_head_repeat_factor(config)
    window_start = int(cache_update_index)
    window_end = window_start + int(seq_len)
    cache_write_index = window_start if cache_write_index is None else int(cache_write_index)
    rope_position_index = window_start if rope_position_index is None else int(rope_position_index)
    rope_position_end = rope_position_index + int(seq_len)
    qk_mode = attention_mode == TRACEABLE_DECODE_ATTENTION_QK_SOFTMAX_MODE
    summary = {
        "mode": attention_mode,
        "attention_output_source": "in_trace_from_q_projection_and_device_kv_cache_window",
        "host_provided_attention_output": False,
        "device_q_projection_contributes": True,
        "device_kv_projection_update_contributes": True,
        "device_kv_cache_read_contributes": True,
        "cache_update_api": cache_update_api,
        "cache_write_index": cache_write_index,
        "cache_write_index_source": "device_tensor"
        if cache_update_api == TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR
        else "host_scalar",
        "cache_write_index_dynamic": cache_update_api == TRACEABLE_DECODE_CACHE_UPDATE_DEVICE_TENSOR,
        "cache_read_window_dynamic": False,
        "cache_read_window_position_source": "static_python_slice_bounds",
        "cache_window": {
            "start": window_start,
            "end_exclusive": window_end,
            "length": int(seq_len),
            "rows": [window_start, window_end],
            "logical_decode_row": int(cache_update_index),
            "updated_row_is_first_window_row": True,
            "static_padding_rows": max(int(seq_len) - 1, 0),
        },
        "cache_expand": {
            "kv_output_dim": _kv_output_dim(config),
            "attention_output_dim": int(config.num_attention_heads) * int(config.head_dim),
            "repeat_factor": head_repeat_factor if qk_mode else width_repeat_factor,
            "repeat_axis": "attention_heads" if qk_mode else "attention_width",
            "op": (
                "ttnn.repeat(..., Shape([1, num_attention_heads, 1, 1]))"
                if qk_mode
                else "ttnn.repeat(..., Shape([1, 1, 1, repeat_factor]))"
            ),
        },
        "kv_source": {
            "key_source": "compressed_kv_projection_cache_window_split_nope_rope_with_rope_rotated",
            "value_source": "compressed_kv_projection_cache_window",
            "key_value_share_same_cache_slice": True,
            "key_value_identical_in_trace": not qk_mode,
            "explicit_kv_tensors_in_trace": qk_mode,
            "kv_split_in_trace": qk_mode,
            "kv_split_kind": "device_resident_explicit_key_and_value_from_single_compressed_cache_window"
            if qk_mode
            else "none_legacy_q_plus_kv_blend",
            "true_kv_split_in_trace": False,
            "true_kv_split_blocker": (
                "the current compressed cache width equals one head_dim and does not expose separate DeepSeek value "
                "projection channels; the trace now creates distinct K and V tensors from that cache window"
            ),
            "num_key_value_heads": int(config.num_key_value_heads),
            "head_repeat_factor": head_repeat_factor,
        },
        "rope": {
            "q_rope_split_in_trace": qk_mode,
            "k_rope_split_in_trace": qk_mode,
            "kv_rope_split_in_trace": qk_mode,
            "q_rope_rotation_in_trace": qk_mode,
            "k_rope_rotation_in_trace": qk_mode,
            "rope_in_trace": qk_mode,
            "qk_rope_head_dim": int(config.qk_rope_head_dim),
            "nope_head_dim": int(config.head_dim) - int(config.qk_rope_head_dim),
            "position_rows": [rope_position_index, rope_position_end],
            "position_dynamic": rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR,
            "position_source": "mutable_device_embedding_indices"
            if rope_position_api == TRACEABLE_DECODE_ROPE_POSITION_DEVICE_TENSOR
            else "static_host_materialized_tables",
            "position_api": rope_position_api,
            "frequency_source": "DeepSeek V4 Flash compressed-layer RoPE frequencies from compress_rope_theta"
            if int(config.compress_ratios[int(layer)]) != 0
            else "DeepSeek V4 Flash base RoPE frequencies",
            "status": "traceable_q_and_k_rope_fixed_window" if qk_mode else "not_used_in_legacy_blend_mode",
        },
        "sparse_compressed_tokens": {
            "contributed": True,
            "source": "device-resident compressed K/V cache window with row 0 updated from real wkv/kv_norm projection",
            "real_sparse_indexer_selected_tokens": False,
        },
        "compressed_token_contribution": {
            "contributed": True,
            "source": "device-resident compressed K/V cache window",
            "updated_row_from_real_kv_projection": True,
        },
        "softmax": {
            "qk_scores_in_trace": qk_mode,
            "fixed_window_softmax_in_trace": qk_mode,
            "attention_sink_softmax_in_trace": False,
            "value_reduction_in_trace": qk_mode,
            "context_in_trace": qk_mode,
        },
        "qk_scores": {
            "produced_in_trace": qk_mode,
            "shape": [1, int(config.num_attention_heads), int(seq_len), int(seq_len)] if qk_mode else None,
            "scale": f"1/sqrt(head_dim={int(config.head_dim)})" if qk_mode else None,
        },
        "context": {
            "produced_in_trace": qk_mode,
            "shape": [1, 1, int(seq_len), int(config.num_attention_heads) * int(config.head_dim)],
            "value_source": "unrotated_compressed_kv_projection_cache_window_repeated_across_attention_heads"
            if qk_mode
            else "expanded_kv_cache_window_blended_with_q_output",
        },
        "exact_sparse_attention_blockers": [
            "dynamic per-token top-k cache gather is still represented by host fallback helpers",
            "attention-sink semantics over [selected cache rows + sink] are not in this protected path",
            "true K/V split is not in this trace slice; K and V are explicit tensors but both derive from the compressed KV projection cache window",
        ],
    }
    if not qk_mode:
        summary["exact_sparse_attention_blockers"].append(
            "legacy mode does not compute QK scores, softmax, or value reduction"
        )
    return summary


def _guard_status(trace_capture: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(trace_capture["guard_enabled"]),
        "ttnn_to_torch_guarded": bool(trace_capture["ttnn_to_torch_guarded"]),
        "guarded_symbol_count": len(trace_capture["guarded_symbols"]),
        "host_boundaries_inside_trace": list(trace_capture["host_boundaries_inside_trace"]),
    }


def _router_mode(*, config: DeepSeekV4FlashConfig, weights: TraceableDecodeWeights) -> str:
    if weights.router_tid2eid is not None:
        return "device_gate_scoring_static_tid2eid_indices_static_dispatch"
    return "device_gate_scoring_topk_route_weights_static_dispatch"


def _router_trace_summary(
    route_plan: TraceableDecodeRoutePlan,
    *,
    config: DeepSeekV4FlashConfig,
    weights: TraceableDecodeWeights,
) -> dict[str, Any]:
    uses_tid2eid = weights.router_tid2eid is not None
    selected_indices = [int(value) for value in route_plan.selected_indices.reshape(-1).tolist()]
    full_indices = [int(value) for value in route_plan.router_indices.reshape(-1).tolist()]
    static_token_rows = _route_plan_static_token_rows(route_plan)
    return {
        "mode": _router_mode(config=config, weights=weights),
        "gate_matmul_in_trace": True,
        "scoring_in_trace": True,
        "bias_in_trace": weights.router_bias is not None,
        "tid2eid_static_indices_in_trace": uses_tid2eid,
        "topk_in_trace": not uses_tid2eid,
        "indices_dynamic_in_trace": not uses_tid2eid,
        "route_weights_in_trace": True,
        "route_weight_normalization_in_trace": str(config.scoring_func) != "softmax",
        "route_weight_decode_row_mask_in_trace": True,
        "expert_ids_dynamic": False,
        "expert_dispatch": "static_preflight",
        "expert_dispatch_dynamic_in_trace": False,
        "expert_dispatch_static_in_trace": True,
        "selected_expert_ids": selected_indices,
        "expected_expert_ids": selected_indices,
        "expected_full_topk_expert_ids": full_indices,
        "selected_topk_prefix": int(route_plan.topk_prefix),
        "full_topk": int(route_plan.full_topk),
        "topk_prefix_is_full": int(route_plan.topk_prefix) == int(route_plan.full_topk),
        "scoring_func": str(config.scoring_func),
        "routed_scaling_factor": float(config.routed_scaling_factor),
        "router_logits_shape": [1, 1, static_token_rows, int(config.n_routed_experts)],
        "router_topk_shape": [1, 1, static_token_rows, int(route_plan.full_topk)],
        "selected_route_weights_shape": [
            1,
            1,
            static_token_rows,
            int(route_plan.topk_prefix),
        ],
        "accuracy_scope": "logical_decode_row",
        "full_static_rows_reported_as_diagnostics": True,
        "static_dispatch_note": (
            "device router weights are consumed by the selected expert slots, but the expert modules are still "
            "chosen by the host preflight plan because dynamic expert dispatch is not wired into this trace path"
        ),
        "topk_blockers": []
        if not uses_tid2eid
        else [
            "hash-routed layers select experts from tid2eid[input_ids]; this path uploads static indices for the "
            "fixed decode token instead of performing a device-side dynamic metadata lookup"
        ],
        "next_dispatch_step": (
            "feed router_topk_indices into a device-side expert dispatch/remap path and instantiate experts from "
            "device-selected ids instead of the host preflight plan"
        ),
    }


def _route_plan_summary(route_plan: TraceableDecodeRoutePlan, *, config: DeepSeekV4FlashConfig) -> dict[str, Any]:
    selected_weights = [float(value) for value in route_plan.selected_weights.reshape(-1).float().tolist()]
    selected_indices = [int(value) for value in route_plan.selected_indices.reshape(-1).tolist()]
    full_weights = [float(value) for value in route_plan.router_weights.reshape(-1).float().tolist()]
    full_indices = [int(value) for value in route_plan.router_indices.reshape(-1).tolist()]
    topk_in_trace = route_plan.input_ids is None
    return {
        "selection_boundary": "host_preflight_static_dispatch_device_router_weights",
        "router_mode": "device_gate_scoring_topk_route_weights_static_dispatch"
        if topk_in_trace
        else "device_gate_scoring_static_tid2eid_indices_static_dispatch",
        "decode_token_index": int(route_plan.decode_token_index),
        "logical_decode_token_only": True,
        "static_token_rows_in_trace": int(next(iter(route_plan.per_expert_route_weight.values())).shape[1]),
        "padded_static_rows_have_zero_routed_weight": True,
        "full_topk": int(route_plan.full_topk),
        "topk_prefix_limit": int(route_plan.topk_prefix),
        "topk_prefix_is_full": int(route_plan.topk_prefix) == int(route_plan.full_topk),
        "full_topk_mode": int(route_plan.topk_prefix) == int(route_plan.full_topk),
        "selected_expert_ids": selected_indices,
        "selected_expert_count": len(selected_indices),
        "executed_expert_ids": selected_indices,
        "executed_expert_count": len(selected_indices),
        "selected_route_weights": selected_weights,
        "full_router_indices_for_decode_token": full_indices,
        "full_router_weights_for_decode_token": full_weights,
        "route_weights_device_resident_inside_trace": True,
        "route_weights_source": "device_router_topk"
        if topk_in_trace
        else "device_router_scoring_static_tid2eid_indices",
        "device_topk_indices_in_trace": topk_in_trace,
        "dynamic_expert_dispatch_in_trace": False,
        "static_expert_dispatch": True,
        "router_scoring_func": str(config.scoring_func),
        "routed_scaling_factor": float(config.routed_scaling_factor),
        "input_ids": _optional_tensor_summary(route_plan.input_ids),
        "limitation": (
            "expert ids still come from a host preflight plan for static expert module construction; "
            "supported router scoring/top-k/route-weight production runs on device inside the protected trace"
        ),
    }


def _route_plan_static_token_rows(route_plan: TraceableDecodeRoutePlan) -> int:
    if not route_plan.per_expert_route_weight:
        raise ValueError("traceable decode route plan must contain at least one selected expert route weight")
    token_rows = {int(value.shape[1]) for value in route_plan.per_expert_route_weight.values()}
    if len(token_rows) != 1:
        raise ValueError(f"selected expert route weights disagree on static token rows: {sorted(token_rows)}")
    return next(iter(token_rows))


def _optional_tensor_summary(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    return None if tensor is None else _tensor_summary(tensor)


def _traceable_accuracy_items(reference: Mapping[str, torch.Tensor]):
    informational_router_rows = {
        "router_topk_selection_scores",
        "router_topk_indices",
        "router_topk_route_scores",
        "router_route_weights",
        "router_selected_route_weights",
    }
    for name, expected in reference.items():
        if name in informational_router_rows:
            continue
        yield name, expected


def _traceable_accuracy_required_for_pass(name: str) -> bool:
    router_dispatch_diagnostics = {
        "router_decode_topk_indices",
    }
    return name not in router_dispatch_diagnostics


def _traceable_accuracy_summary(
    name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    pcc_threshold: float,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    projection_outputs = {"attention_projected", "post_attention_residual", "combined_ffn_output", "residual_output"}
    attention_split_diagnostics = {
        "attention_q_heads",
        "attention_q_heads_norm",
        "attention_q_nope",
        "attention_q_rope",
        "attention_q_rope_rotated",
    }
    local_rtol = max(float(rtol), ATTENTION_OUTPUT_PROJECTION_RTOL) if name in projection_outputs else float(rtol)
    local_atol = max(float(atol), ATTENTION_OUTPUT_PROJECTION_ATOL) if name in projection_outputs else float(atol)
    if name in attention_split_diagnostics:
        local_atol = max(local_atol, 2.5e-1)
    summary = _accuracy_summary(
        expected,
        actual,
        pcc_threshold=pcc_threshold,
        rtol=local_rtol,
        atol=local_atol,
    )
    if name == "attention_probs" and summary["allclose"]:
        summary["passed"] = True
        summary["pcc_note"] = (
            "softmax probabilities are reported with PCC, but pass/fail is gated by allclose because "
            "near-uniform probability rows can have low-variance PCC despite small absolute error"
        )
    if name in {"router_decode_route_weights", "router_decode_selected_route_weights"} and summary["allclose"]:
        summary["passed"] = True
        summary["pcc_note"] = (
            "decode-row router weights are reported with PCC, but pass/fail is gated by allclose because "
            "small top-k weight vectors can have low-variance PCC despite small absolute error"
        )
    if name == "router_decode_topk_indices":
        summary["diagnostic_note"] = (
            "device top-k indices are reported against the torch preflight reference, but static expert dispatch "
            "does not consume these ids yet; this metric is not required for smoke pass/fail"
        )
    if name in projection_outputs:
        summary[
            "tolerance_note"
        ] = "traceable decode matmul/SwiGLU path uses relaxed absolute tolerance; PCC remains enforced"
    if name in attention_split_diagnostics:
        summary[
            "tolerance_note"
        ] = "internal Q split/RoPE diagnostic uses relaxed absolute tolerance; PCC and final attention outputs remain enforced"
    return summary


def _replay_activation(activation: torch.Tensor) -> torch.Tensor:
    token_scale = torch.linspace(1.05, 0.95, steps=activation.shape[-2], dtype=torch.float32).reshape(1, 1, -1, 1)
    return (activation.float().flip(-2) * token_scale - 0.03125).to(torch.bfloat16).contiguous()


def _decode_step_activation(activation: torch.Tensor, step: int) -> torch.Tensor:
    if int(step) == 0:
        return activation.contiguous()
    token_scale = torch.linspace(0.97, 1.03, steps=activation.shape[-2], dtype=torch.float32).reshape(1, 1, -1, 1)
    shifted = activation.float().roll(shifts=int(step), dims=-2)
    return (shifted * token_scale + (0.015625 * int(step))).to(torch.bfloat16).contiguous()


def deterministic_traceable_kv_cache_seed(
    *,
    cache_len: int,
    kv_output_dim: int,
    cache_update_index: int,
    seq_len: int,
    decode_steps: int = DEFAULT_TRACEABLE_DECODE_STEPS,
) -> torch.Tensor:
    if cache_len <= 0:
        raise ValueError(f"cache_len must be positive, got {cache_len}")
    if kv_output_dim <= 0:
        raise ValueError(f"kv_output_dim must be positive, got {kv_output_dim}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if decode_steps <= 0:
        raise ValueError(f"decode_steps must be positive, got {decode_steps}")
    if cache_update_index < 0 or cache_update_index + int(decode_steps) - 1 + seq_len > cache_len:
        raise ValueError(
            f"last cache window [{cache_update_index + int(decode_steps) - 1}, "
            f"{cache_update_index + int(decode_steps) - 1 + seq_len}) must fit cache_len {cache_len}"
        )

    values = torch.linspace(-0.125, 0.125, steps=cache_len * kv_output_dim, dtype=torch.float32)
    cache = values.reshape(1, 1, cache_len, kv_output_dim)
    row_scale = torch.linspace(0.85, 1.15, steps=cache_len, dtype=torch.float32).reshape(1, 1, cache_len, 1)
    cache = cache * row_scale
    cache[:, :, int(cache_update_index) : int(cache_update_index) + int(decode_steps), :] = 0.0
    return cache.to(torch.bfloat16).contiguous()


def _validate_ttnn_hidden_states(hidden_states, *, hidden_size: int) -> None:
    shape = tuple(hidden_states.shape)
    if len(shape) != 4 or shape[:2] != (1, 1):
        raise ValueError(f"hidden_states must have shape [1, 1, tokens, hidden], got {shape}")
    if int(shape[-1]) != int(hidden_size):
        raise ValueError(f"hidden_states hidden dim must be {hidden_size}, got {shape[-1]}")
    if int(shape[-2]) <= 0:
        raise ValueError("hidden_states must contain at least one token")


def _validate_activation(activation: torch.Tensor, *, hidden_size: int) -> None:
    if activation.ndim != 4 or tuple(activation.shape[:2]) != (1, 1):
        raise ValueError(f"activation must have shape [1, 1, tokens, hidden], got {tuple(activation.shape)}")
    if int(activation.shape[-1]) != int(hidden_size):
        raise ValueError(f"activation hidden size must be {hidden_size}, got {activation.shape[-1]}")
    if int(activation.shape[-2]) <= 0:
        raise ValueError("activation must contain at least one token")


def _validate_kv_cache_initial(kv_cache_initial: torch.Tensor, *, cache_len: int, kv_output_dim: int) -> None:
    expected_shape = (1, 1, int(cache_len), int(kv_output_dim))
    if tuple(kv_cache_initial.shape) != expected_shape:
        raise ValueError(f"kv_cache_initial must have shape {expected_shape}, got {tuple(kv_cache_initial.shape)}")


def _validate_traceable_decode_router_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
) -> None:
    prefix = f"layers.{layer}"
    gate_key = f"{prefix}.ffn.gate.weight"
    if gate_key not in tensors:
        raise KeyError(f"Missing required router tensor {gate_key!r}")
    expected_gate_shape = (int(config.n_routed_experts), int(config.hidden_size))
    if tuple(tensors[gate_key].shape) != expected_gate_shape:
        raise ValueError(f"Expected {gate_key} shape {expected_gate_shape}, got {tuple(tensors[gate_key].shape)}")

    bias_key = f"{prefix}.ffn.gate.bias"
    tid2eid_key = f"{prefix}.ffn.gate.tid2eid"
    has_bias = bias_key in tensors
    has_tid2eid = tid2eid_key in tensors
    if has_bias == has_tid2eid:
        raise ValueError(f"Expected exactly one of {bias_key!r} or {tid2eid_key!r}")
    if has_bias and tuple(tensors[bias_key].shape) != (int(config.n_routed_experts),):
        raise ValueError(
            f"Expected {bias_key} shape {(int(config.n_routed_experts),)}, got {tuple(tensors[bias_key].shape)}"
        )
    if has_tid2eid:
        expected_tid2eid_shape = (int(config.vocab_size), int(config.num_experts_per_tok))
        if tuple(tensors[tid2eid_key].shape) != expected_tid2eid_shape:
            raise ValueError(
                f"Expected {tid2eid_key} shape {expected_tid2eid_shape}, got {tuple(tensors[tid2eid_key].shape)}"
            )


def _traceable_decode_input_ids(tid2eid: torch.Tensor | None, *, seq_len: int) -> torch.Tensor | None:
    if tid2eid is None:
        return None
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    return torch.zeros((1, int(seq_len)), dtype=torch.long)


def _validate_smoke_args(
    *,
    layer: int,
    seq_len: int,
    max_tensors: int,
    max_bytes: int,
    trace_region_size: int,
    cache_len: int,
    cache_update_index: int | None,
    decode_steps: int,
    routed_topk_prefix: int | None,
    pcc: float,
) -> None:
    if layer < 0:
        raise ValueError(f"layer must be non-negative, got {layer}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if max_tensors <= 0:
        raise ValueError(f"max_tensors must be positive, got {max_tensors}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    if trace_region_size <= 0:
        raise ValueError(f"trace_region_size must be positive, got {trace_region_size}")
    if cache_len <= 0:
        raise ValueError(f"cache_len must be positive, got {cache_len}")
    if decode_steps <= 0:
        raise ValueError(f"decode_steps must be positive, got {decode_steps}")
    if cache_update_index is not None:
        update_index = _resolve_cache_update_index(
            seq_len=seq_len, cache_len=cache_len, cache_update_index=cache_update_index
        )
    elif seq_len >= cache_len:
        raise ValueError(f"default cache_update_index seq_len={seq_len} must be less than cache_len {cache_len}")
    else:
        update_index = int(seq_len)
    last_update_index = update_index + int(decode_steps) - 1
    if last_update_index + int(seq_len) > int(cache_len):
        raise ValueError(
            f"last fixed attention cache window [{last_update_index}, {last_update_index + int(seq_len)}) "
            f"exceeds cache_len {cache_len}"
        )
    if routed_topk_prefix is not None and routed_topk_prefix <= 0:
        raise ValueError(f"routed_topk_prefix must be positive when provided, got {routed_topk_prefix}")
    if not 0.0 <= pcc <= 1.0:
        raise ValueError(f"pcc must be in [0, 1], got {pcc}")


def _load_missing_tensors(
    index: RealCheckpointTensorIndex,
    tensors: dict[str, torch.Tensor],
    metadata: list[TensorMetadata],
    keys: Sequence[str],
    *,
    max_tensors: int,
    max_bytes: int,
) -> tuple[dict[str, torch.Tensor], list[TensorMetadata]]:
    missing_keys = [key for key in keys if key not in tensors]
    if not missing_keys:
        return tensors, metadata
    used_bytes = sum(item.nbytes for item in metadata)
    loaded_tensors, loaded_metadata = index.load_tensors(
        missing_keys,
        max_tensors=max_tensors - len(metadata),
        max_bytes=max_bytes - used_bytes,
    )
    tensors.update(loaded_tensors)
    return tensors, [*metadata, *loaded_metadata]


def _unique_keys(keys: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


if __name__ == "__main__":
    main()
