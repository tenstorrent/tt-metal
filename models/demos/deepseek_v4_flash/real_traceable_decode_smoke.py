# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib
import json
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
    append, grouped attention output projection, post-attention residual, and
    preselected routed/shared FFN stepping stone:
    ``hidden -> attn_norm -> wq_a -> q_norm -> wq_b``,
    ``attn_norm -> wkv -> kv_norm -> update_cache``,
    ``attention_output -> grouped wo_a -> wo_b -> residual``, and
    ``post_attention_residual -> ffn_norm -> routed/shared experts -> residual``.
    """

    def __init__(
        self,
        *,
        device,
        weights: TraceableDecodeWeights,
        route_plan: TraceableDecodeRoutePlan,
        config: DeepSeekV4FlashConfig,
        cache_len: int,
        cache_update_index: int,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        self.device = device
        self.config = config
        self.dtype = dtype
        self.memory_config = memory_config
        self.cache_len = int(cache_len)
        self.cache_update_index = int(cache_update_index)
        self.route_plan = route_plan
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
        self.routed_route_weights: dict[int, object] = {}
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
            self.routed_route_weights[expert] = _to_tt_route_weight(
                route_plan.per_expert_route_weight[expert],
                intermediate_size=routed_module.intermediate_size,
                device=device,
                dtype=dtype,
                memory_config=memory_config,
            )
        self.kv_output_dim = _kv_output_dim(config)
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
        self.kv_cache = _to_tt_kv_cache(
            cache_len=self.cache_len,
            kv_output_dim=self.kv_output_dim,
            device=device,
            dtype=dtype,
            memory_config=memory_config,
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

    def __call__(self, hidden_states, attention_output) -> dict[str, object]:
        _validate_ttnn_hidden_states(hidden_states, hidden_size=int(self.config.hidden_size))
        _validate_ttnn_attention_output(
            attention_output, q_output_dim=int(self.config.num_attention_heads) * int(self.config.head_dim)
        )
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
        kv_update = ttnn.to_memory_config(
            kv_output,
            _kv_update_memory_config(device=self.device, token_rows=int(kv_output.shape[-2]), width=self.kv_output_dim),
        )
        self.kv_cache = ttnn.update_cache(self.kv_cache, kv_update, self.cache_update_index)
        attention_projected = self.attention.project_output(attention_output)
        post_attention_residual = ttnn.add(hidden_states, attention_projected, memory_config=self.memory_config)
        ffn_norm_output = ttnn.rms_norm(
            post_attention_residual,
            weight=self.ffn_norm,
            epsilon=float(self.config.rms_norm_eps),
            memory_config=self.memory_config,
        )
        shared_output = self.shared_expert(ffn_norm_output)
        routed_expert_outputs = {
            f"routed_expert_{expert}_output": routed_module(
                ffn_norm_output,
                route_weight=self.routed_route_weights[expert],
            )
            for expert, routed_module in self.routed_experts.items()
        }
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
            "attention_projected": attention_projected,
            "post_attention_residual": post_attention_residual,
            "ffn_norm_output": ffn_norm_output,
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
        routed_topk_prefix=routed_topk_prefix,
        pcc=pcc,
    )
    cache_update_index = _resolve_cache_update_index(
        seq_len=seq_len,
        cache_len=cache_len,
        cache_update_index=cache_update_index,
    )
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
    attention_output = deterministic_traceable_attention_output(
        q_output_dim=int(config.num_attention_heads) * int(config.head_dim),
        seq_len=seq_len,
    )
    replay_activation = _replay_activation(activation)
    replay_attention_output = _replay_attention_output(attention_output)
    preliminary_reference = build_torch_traceable_decode_subpath_reference(
        weights,
        config=config,
        activation=activation,
        attention_output=attention_output,
        cache_len=cache_len,
        cache_update_index=cache_update_index,
        route_plan=None,
    )
    route_plan = build_traceable_decode_route_plan(
        weights,
        config=config,
        seq_len=seq_len,
        ffn_norm_output=preliminary_reference["ffn_norm_output"],
        routed_topk_prefix=routed_topk_prefix,
    )
    if route_plan.selected_expert_ids:
        index = RealCheckpointTensorIndex.from_snapshot(snapshot_dir)
        routed_expert_keys = [
            key
            for expert in route_plan.selected_expert_ids
            for key in layer_expert_mlp_keys(index, layer=layer, expert=expert)
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
            selected_experts=route_plan.selected_expert_ids,
        )
    reference = build_torch_traceable_decode_subpath_reference(
        weights,
        config=config,
        activation=activation,
        attention_output=attention_output,
        cache_len=cache_len,
        cache_update_index=cache_update_index,
        route_plan=route_plan,
    )
    replay_reference = build_torch_traceable_decode_subpath_reference(
        weights,
        config=config,
        activation=replay_activation,
        attention_output=replay_attention_output,
        cache_len=cache_len,
        cache_update_index=cache_update_index,
        route_plan=route_plan,
    )
    metadata_groups = _metadata_groups(metadata, keys)
    result = _base_result(
        snapshot_dir=snapshot_dir,
        layer=layer,
        seq_len=seq_len,
        cache_len=cache_len,
        cache_update_index=cache_update_index,
        config=config,
        metadata=metadata,
        metadata_groups=metadata_groups,
        weights=weights,
        activation=activation,
        attention_output=attention_output,
        replay_activation=replay_activation,
        replay_attention_output=replay_attention_output,
        route_plan=route_plan,
        reference=reference,
        replay_reference=replay_reference,
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
        result["passed"] = True
        return result

    if seq_len % SHARED_EXPERT_TTNN_TILE_MULTIPLE != 0:
        raise ValueError(f"TTNN traceable decode seq_len must be a multiple of 32, got {seq_len}")

    ttnn_outputs, trace_info = _run_ttnn_traceable_decode_subpath(
        weights,
        route_plan=route_plan,
        config=config,
        activation=activation,
        attention_output=attention_output,
        replay_activation=replay_activation,
        replay_attention_output=replay_attention_output,
        device_id=device_id,
        trace_region_size=trace_region_size,
        cache_len=cache_len,
        cache_update_index=cache_update_index,
    )
    result["mode"] = "ttnn-trace"
    result["device_id"] = int(device_id)
    result["trace_capture"].update(trace_info)
    result["trace_capture_attempted"] = bool(result["trace_capture"]["attempted"])
    result["trace_capture_passed"] = bool(result["trace_capture"]["capture_passed"])
    result["trace_execute_replay_passed"] = bool(result["trace_capture"]["execute_replay_passed"])
    result["guard_status"] = _guard_status(result["trace_capture"])
    result["ttnn_ops"] = [
        "ttnn.rms_norm(attn_norm)",
        "ttnn.linear(wq_a)",
        "ttnn.rms_norm(q_norm)",
        "ttnn.linear(wq_b)",
        "ttnn.linear(wkv)",
        "ttnn.rms_norm(kv_norm)",
        "ttnn.to_memory_config(kv_update_height_sharded)",
        "ttnn.update_cache(kv_projection_cache)",
        "ttnn.slice(attention_output_group_0..N)",
        "ttnn.linear(grouped_wo_a_group_0..N)",
        "ttnn.concat(grouped_wo_a_rank)",
        "ttnn.linear(wo_b)",
        "ttnn.add(hidden,attention_projected)",
        "ttnn.rms_norm(ffn_norm)",
        "ttnn.linear(shared_w1)",
        "ttnn.linear(shared_w3)",
        "ttnn.mul(silu(shared_gate),shared_up)",
        "ttnn.linear(shared_w2)",
        "ttnn.linear(routed_w1_selected_topk_prefix)",
        "ttnn.linear(routed_w3_selected_topk_prefix)",
        "ttnn.mul(silu(routed_gate),routed_up)",
        "ttnn.mul(routed_hidden,preselected_route_weight)",
        "ttnn.linear(routed_w2_selected_topk_prefix)",
        "ttnn.add(routed_expert_outputs)",
        "ttnn.add(shared_output,routed_output)",
        "ttnn.add(post_attention_residual,combined_ffn_output)",
    ]
    result["trace_capture"]["traced_operations"] = list(result["ttnn_ops"])
    result["ttnn"] = {name: _tensor_summary(value) for name, value in ttnn_outputs.items()}
    result["accuracy"] = {
        name: _traceable_accuracy_summary(
            name,
            expected,
            ttnn_outputs[name],
            pcc_threshold=pcc,
            rtol=rtol,
            atol=atol,
        )
        for name, expected in replay_reference.items()
    }
    result["passed"] = bool(
        result["trace_capture"]["attempted"]
        and result["trace_capture"]["capture_passed"]
        and result["trace_capture"]["execute_replay_passed"]
        and all(item["passed"] for item in result["accuracy"].values())
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
    activation: torch.Tensor,
    attention_output: torch.Tensor,
    cache_len: int,
    cache_update_index: int,
    route_plan: TraceableDecodeRoutePlan | None,
) -> dict[str, torch.Tensor]:
    _validate_activation(activation, hidden_size=int(config.hidden_size))
    _validate_attention_output(attention_output, q_output_dim=int(config.num_attention_heads) * int(config.head_dim))
    attn_norm_output = rms_norm(
        activation[:, 0],
        weights.attn_norm,
        eps=float(config.rms_norm_eps),
    ).unsqueeze(1)
    q_rank_linear = F.linear(attn_norm_output[:, 0].float(), weights.attention.wq_a.float()).to(torch.bfloat16)
    q_rank_norm = rms_norm(q_rank_linear, weights.attention.q_norm, eps=float(config.rms_norm_eps)).unsqueeze(1)
    q_output = F.linear(q_rank_norm[:, 0].float(), weights.attention.wq_b.float()).unsqueeze(1)
    kv_linear = F.linear(attn_norm_output[:, 0].float(), weights.kv.wkv.float()).to(torch.bfloat16)
    kv_output = rms_norm(kv_linear, weights.kv.kv_norm, eps=float(config.rms_norm_eps)).unsqueeze(1)
    kv_cache = torch.zeros((1, 1, int(cache_len), _kv_output_dim(config)), dtype=torch.bfloat16)
    kv_cache[:, :, int(cache_update_index) : int(cache_update_index) + 1, :] = kv_output[:, :, :1, :]

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
        for expert in route_plan.selected_expert_ids:
            if expert not in weights.routed_experts:
                raise KeyError(f"Missing routed expert weights for selected expert {expert}")
            expert_weights = weights.routed_experts[expert]
            route_weight = route_plan.per_expert_route_weight[expert].reshape(-1, 1)
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
        "attention_projected": attention_projected.to(torch.bfloat16),
        "post_attention_residual": post_attention_residual.to(torch.bfloat16),
        "ffn_norm_output": ffn_norm_output.to(torch.bfloat16),
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
    route_plan: TraceableDecodeRoutePlan,
    config: DeepSeekV4FlashConfig,
    activation: torch.Tensor,
    attention_output: torch.Tensor,
    replay_activation: torch.Tensor,
    replay_attention_output: torch.Tensor,
    device_id: int,
    trace_region_size: int,
    cache_len: int,
    cache_update_index: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    device = ttnn.open_device(
        device_id=int(device_id),
        num_command_queues=1,
        trace_region_size=int(trace_region_size),
    )
    trace_id = None
    try:
        module = TtTraceableDecodeSubpath(
            device=device,
            weights=weights,
            route_plan=route_plan,
            config=config,
            cache_len=cache_len,
            cache_update_index=cache_update_index,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_input = ttnn.allocate_tensor_on_device(
            ttnn.Shape(tuple(activation.shape)),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
        )
        tt_attention_output = ttnn.allocate_tensor_on_device(
            ttnn.Shape(tuple(attention_output.shape)),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
        )
        _copy_activation_to_device(activation, tt_input)
        _copy_activation_to_device(attention_output, tt_attention_output)
        module(tt_input, tt_attention_output)
        ttnn.synchronize_device(device)

        with TraceableDecodeHostGuard() as guard:
            trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            output_tensors = module(tt_input, tt_attention_output)
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
        ttnn.synchronize_device(device)

        _copy_activation_to_device(replay_activation, tt_input)
        _copy_activation_to_device(replay_attention_output, tt_attention_output)
        ttnn.synchronize_device(device)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(device)
        outputs = {name: ttnn.to_torch(tensor).contiguous() for name, tensor in output_tensors.items()}
        trace_info = {
            "attempted": True,
            "capture_passed": True,
            "execute_replay_attempted": True,
            "execute_replay_passed": True,
            "trace_id_allocated": True,
            "guard_enabled": True,
            "guarded_symbols": guard.guarded_labels,
            "ttnn_to_torch_guarded": "ttnn.to_torch" in guard.guarded_labels,
            "host_boundaries_inside_trace": [],
        }
        return outputs, trace_info
    finally:
        if trace_id is not None:
            ttnn.release_trace(device, trace_id)
        ttnn.close_device(device)


def _copy_activation_to_device(activation: torch.Tensor, tt_input) -> None:
    host_tensor = ttnn.from_torch(
        activation.contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn.copy_host_to_device_tensor(host_tensor, tt_input)


def _base_result(
    *,
    snapshot_dir: Path,
    layer: int,
    seq_len: int,
    cache_len: int,
    cache_update_index: int,
    config: DeepSeekV4FlashConfig,
    metadata: Sequence[TensorMetadata],
    metadata_groups: Mapping[str, Sequence[TensorMetadata]],
    weights: TraceableDecodeWeights,
    activation: torch.Tensor,
    attention_output: torch.Tensor,
    replay_activation: torch.Tensor,
    replay_attention_output: torch.Tensor,
    route_plan: TraceableDecodeRoutePlan,
    reference: Mapping[str, torch.Tensor],
    replay_reference: Mapping[str, torch.Tensor],
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
    routed_expert_ids_loaded = [int(expert) for expert in weights.routed_experts]
    routed_expert_ids_executed = list(route_plan.selected_expert_ids)
    excluded_from_trace = [
        "K/V RoPE split and final sparse-attention cache read path",
        "host sparse-attention gather/softmax/reduction",
        "real sparse-attention output production; deterministic attention tensor is uploaded before trace",
        "router scoring/top-k/hash selection; selected expert ids and route weights are precomputed on host",
        "cache advancement beyond the fixed traced update index",
        "embedding and logits",
    ]
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
        "tensor_sequence_length": int(seq_len),
        "cache_update": {
            "name": "compressed_kv_projection_cache_append",
            "cache_len": int(cache_len),
            "update_index": int(cache_update_index),
            "updated_tokens": 1,
            "input_layout": "[seq=1, heads=1, batch_padded=32, kv_output_dim]",
            "cache_layout": "[batch=1, heads=1, cache_len, kv_output_dim]",
            "device_resident_inside_trace": True,
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
            "inside_trace": [
                "ttnn.rms_norm(attn_norm)",
                "TtAttentionProjection.project_q_rank",
                "TtAttentionProjection.project_q_from_rank",
                "ttnn.linear(wkv)",
                "ttnn.rms_norm(kv_norm)",
                "ttnn.to_memory_config(kv_update_height_sharded)",
                "ttnn.update_cache(kv_projection_cache)",
                "TtAttentionProjection.project_output",
                "ttnn.slice(attention_output_group_0..N)",
                "ttnn.linear(grouped_wo_a_group_0..N)",
                "ttnn.concat(grouped_wo_a_rank)",
                "ttnn.linear(wo_b)",
                "ttnn.add(hidden,attention_projected)",
                "ttnn.rms_norm(ffn_norm)",
                "TtRoutedExpertMLP(selected_topk_prefix)",
                "ttnn.mul(routed_hidden,preselected_route_weight)",
                "ttnn.add(routed_expert_outputs)",
                "TtSharedExpertMLP",
                "ttnn.add(shared_output,routed_output)",
                "ttnn.add(post_attention_residual,combined_ffn_output)",
            ],
            "path": (
                "decode hidden state -> attn_norm/query projection plus K/V projection/cache append; "
                "deterministic device attention tensor -> grouped wo_a/wo_b/post-attention residual; "
                "post-attention residual -> ffn_norm/preselected routed top-k fanout plus shared expert/residual"
            ),
            "logical_decode_token_policy": (
                "the first token is the logical decode token; tensor shape is tile-padded/static for trace replay"
            ),
            "excluded_from_trace": excluded_from_trace,
        },
        "selected_routing": _route_plan_summary(route_plan, config=config),
        "routed_expert_execution": {
            "loaded_expert_ids": routed_expert_ids_loaded,
            "loaded_expert_count": len(routed_expert_ids_loaded),
            "executed_expert_ids": routed_expert_ids_executed,
            "executed_expert_count": len(routed_expert_ids_executed),
            "all_selected_experts_loaded": set(routed_expert_ids_executed).issubset(set(routed_expert_ids_loaded)),
            "full_topk_executed": int(route_plan.topk_prefix) == int(route_plan.full_topk),
        },
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
            "kv_cache_initial": _tensor_summary(torch.zeros((1, 1, int(cache_len), _kv_output_dim(config)))),
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
                "name": "router_topk_pretrace",
                "location": "before protected traceable decode region",
                "description": (
                    "router scoring/top-k runs on host for the logical decode token; the protected trace receives "
                    "fixed selected expert ids and route weights"
                ),
            },
            {
                "name": "route_weight_host_to_device",
                "location": "before trace capture",
                "description": "precomputed route weights are expanded and uploaded as device tensors during module setup",
            },
            {
                "name": "kv_cache_zero_init_host_to_device",
                "location": "before trace capture",
                "description": "the compressed K/V projection cache is zero-initialized on host and uploaded during module setup",
            },
            {
                "name": "activation_host_to_device",
                "location": "before trace capture and before replay",
                "description": "static-shape decode activation is copied into a preallocated device tensor outside the guard",
            },
            {
                "name": "attention_output_host_to_device",
                "location": "before trace capture and before replay",
                "description": (
                    "deterministic sparse-attention output placeholder is copied into a preallocated device tensor "
                    "outside the guard because sparse attention is not yet trace-captured"
                ),
            },
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
            "router_topk_pretrace",
            "route_weight_host_to_device",
            "kv_cache_zero_init_host_to_device",
            "activation_host_to_device",
            "attention_output_host_to_device",
            "trace_output_readback",
        ],
        "reference_ops": [
            "torch.rms_norm_reference(attn_norm)",
            "torch.linear(wq_a)",
            "torch.rms_norm_reference(q_norm)",
            "torch.linear(wq_b)",
            "torch.linear(wkv)",
            "torch.rms_norm_reference(kv_norm)",
            "torch.cache_update_reference",
            "torch.grouped_output_projection_a",
            "torch.linear(wo_b)",
            "torch.add(hidden,attention_projected)",
            "torch.rms_norm_reference(ffn_norm)",
            "torch.routed_swiglu_expert_reference(preselected_topk_prefix)",
            "torch.shared_swiglu_expert_reference",
            "torch.add(shared_output,routed_output)",
            "torch.add(post_attention_residual,combined_ffn_output)",
        ],
        "ttnn_ops": [],
        "inputs": {
            "capture_activation": _tensor_summary(activation),
            "capture_attention_output": _tensor_summary(attention_output),
            "replay_activation": _tensor_summary(replay_activation),
            "replay_attention_output": _tensor_summary(replay_attention_output),
        },
        "reference": {name: _tensor_summary(value) for name, value in reference.items()},
        "replay_reference": {name: _tensor_summary(value) for name, value in replay_reference.items()},
        "ttnn": {},
        "accuracy": {},
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
    device,
    dtype,
    memory_config,
):
    cache = torch.zeros((1, 1, int(cache_len), int(kv_output_dim)), dtype=torch.bfloat16)
    return ttnn.from_torch(
        cache,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )


def _to_tt_route_weight(
    route_weight: torch.Tensor,
    *,
    intermediate_size: int,
    device,
    dtype,
    memory_config,
):
    if route_weight.ndim != 3 or tuple(route_weight.shape[:1]) != (1,) or int(route_weight.shape[-1]) != 1:
        raise ValueError(f"route_weight must have shape [1, tokens, 1], got {tuple(route_weight.shape)}")
    expanded = (
        route_weight.reshape(1, 1, int(route_weight.shape[-2]), 1)
        .expand(1, 1, int(route_weight.shape[-2]), int(intermediate_size))
        .contiguous()
        .to(torch.bfloat16)
    )
    return ttnn.from_torch(
        expanded,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )


def _sum_ttnn_tensors(tensors, *, memory_config):
    values = list(tensors)
    if not values:
        raise ValueError("at least one routed expert output is required")
    result = values[0]
    for value in values[1:]:
        result = ttnn.add(result, value, memory_config=memory_config)
    return result


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


def _kv_output_dim(config: DeepSeekV4FlashConfig) -> int:
    return int(config.num_key_value_heads) * int(config.head_dim)


def _resolve_cache_update_index(*, seq_len: int, cache_len: int, cache_update_index: int | None) -> int:
    update_index = int(seq_len) if cache_update_index is None else int(cache_update_index)
    if update_index < 0:
        raise ValueError(f"cache_update_index must be non-negative, got {update_index}")
    if update_index >= int(cache_len):
        raise ValueError(f"cache_update_index {update_index} must be less than cache_len {cache_len}")
    return update_index


def _guard_status(trace_capture: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(trace_capture["guard_enabled"]),
        "ttnn_to_torch_guarded": bool(trace_capture["ttnn_to_torch_guarded"]),
        "guarded_symbol_count": len(trace_capture["guarded_symbols"]),
        "host_boundaries_inside_trace": list(trace_capture["host_boundaries_inside_trace"]),
    }


def _route_plan_summary(route_plan: TraceableDecodeRoutePlan, *, config: DeepSeekV4FlashConfig) -> dict[str, Any]:
    selected_weights = [float(value) for value in route_plan.selected_weights.reshape(-1).float().tolist()]
    selected_indices = [int(value) for value in route_plan.selected_indices.reshape(-1).tolist()]
    full_weights = [float(value) for value in route_plan.router_weights.reshape(-1).float().tolist()]
    full_indices = [int(value) for value in route_plan.router_indices.reshape(-1).tolist()]
    return {
        "selection_boundary": "host_pretrace_router_topk",
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
        "router_scoring_func": str(config.scoring_func),
        "routed_scaling_factor": float(config.routed_scaling_factor),
        "input_ids": _optional_tensor_summary(route_plan.input_ids),
        "limitation": (
            "router/top-k is host-computed before trace; this trace replays a fixed selected top-k route plan "
            "for the logical decode token, optionally limited by topk_prefix_limit"
        ),
    }


def _optional_tensor_summary(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    return None if tensor is None else _tensor_summary(tensor)


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
    local_rtol = max(float(rtol), ATTENTION_OUTPUT_PROJECTION_RTOL) if name in projection_outputs else float(rtol)
    local_atol = max(float(atol), ATTENTION_OUTPUT_PROJECTION_ATOL) if name in projection_outputs else float(atol)
    summary = _accuracy_summary(
        expected,
        actual,
        pcc_threshold=pcc_threshold,
        rtol=local_rtol,
        atol=local_atol,
    )
    if name in projection_outputs:
        summary[
            "tolerance_note"
        ] = "traceable decode matmul/SwiGLU path uses relaxed absolute tolerance; PCC remains enforced"
    return summary


def _replay_activation(activation: torch.Tensor) -> torch.Tensor:
    token_scale = torch.linspace(1.05, 0.95, steps=activation.shape[-2], dtype=torch.float32).reshape(1, 1, -1, 1)
    return (activation.float().flip(-2) * token_scale - 0.03125).to(torch.bfloat16).contiguous()


def deterministic_traceable_attention_output(*, q_output_dim: int, seq_len: int) -> torch.Tensor:
    if q_output_dim <= 0:
        raise ValueError(f"q_output_dim must be positive, got {q_output_dim}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    values = torch.linspace(-0.45, 0.55, steps=seq_len * q_output_dim, dtype=torch.float32)
    attention = values.reshape(seq_len, q_output_dim)
    attention = attention - attention.mean(dim=-1, keepdim=True)
    attention = attention * torch.rsqrt(attention.square().mean(dim=-1, keepdim=True) + 1e-6)
    token_offsets = torch.linspace(0.03, -0.03, steps=seq_len, dtype=torch.float32).reshape(seq_len, 1)
    return (attention + token_offsets).reshape(1, 1, seq_len, q_output_dim).to(torch.bfloat16).contiguous()


def _replay_attention_output(attention_output: torch.Tensor) -> torch.Tensor:
    token_scale = torch.linspace(0.9, 1.1, steps=attention_output.shape[-2], dtype=torch.float32).reshape(1, 1, -1, 1)
    return (attention_output.float().roll(shifts=1, dims=-2) * token_scale + 0.015625).to(torch.bfloat16).contiguous()


def _validate_ttnn_hidden_states(hidden_states, *, hidden_size: int) -> None:
    shape = tuple(hidden_states.shape)
    if len(shape) != 4 or shape[:2] != (1, 1):
        raise ValueError(f"hidden_states must have shape [1, 1, tokens, hidden], got {shape}")
    if int(shape[-1]) != int(hidden_size):
        raise ValueError(f"hidden_states hidden dim must be {hidden_size}, got {shape[-1]}")
    if int(shape[-2]) <= 0:
        raise ValueError("hidden_states must contain at least one token")


def _validate_ttnn_attention_output(attention_output, *, q_output_dim: int) -> None:
    shape = tuple(attention_output.shape)
    if len(shape) != 4 or shape[:2] != (1, 1):
        raise ValueError(f"attention_output must have shape [1, 1, tokens, q_output_dim], got {shape}")
    if int(shape[-1]) != int(q_output_dim):
        raise ValueError(f"attention_output width must be {q_output_dim}, got {shape[-1]}")
    if int(shape[-2]) <= 0:
        raise ValueError("attention_output must contain at least one token")


def _validate_activation(activation: torch.Tensor, *, hidden_size: int) -> None:
    if activation.ndim != 4 or tuple(activation.shape[:2]) != (1, 1):
        raise ValueError(f"activation must have shape [1, 1, tokens, hidden], got {tuple(activation.shape)}")
    if int(activation.shape[-1]) != int(hidden_size):
        raise ValueError(f"activation hidden size must be {hidden_size}, got {activation.shape[-1]}")
    if int(activation.shape[-2]) <= 0:
        raise ValueError("activation must contain at least one token")


def _validate_attention_output(attention_output: torch.Tensor, *, q_output_dim: int) -> None:
    if attention_output.ndim != 4 or tuple(attention_output.shape[:2]) != (1, 1):
        raise ValueError(
            f"attention_output must have shape [1, 1, tokens, q_output_dim], got {tuple(attention_output.shape)}"
        )
    if int(attention_output.shape[-1]) != int(q_output_dim):
        raise ValueError(f"attention_output width must be {q_output_dim}, got {attention_output.shape[-1]}")
    if int(attention_output.shape[-2]) <= 0:
        raise ValueError("attention_output must contain at least one token")


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
    if cache_update_index is not None:
        _resolve_cache_update_index(seq_len=seq_len, cache_len=cache_len, cache_update_index=cache_update_index)
    elif seq_len >= cache_len:
        raise ValueError(f"default cache_update_index seq_len={seq_len} must be less than cache_len {cache_len}")
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
