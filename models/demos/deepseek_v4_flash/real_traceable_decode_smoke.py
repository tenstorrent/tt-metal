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
from models.demos.deepseek_v4_flash.cpu_reference import rms_norm, swiglu_expert
from models.demos.deepseek_v4_flash.real_attention_projection_smoke import (
    DEFAULT_ATTENTION_PROJECTION_MAX_BYTES,
    DEFAULT_ATTENTION_PROJECTION_MAX_TENSORS,
    _accuracy_summary,
    _metadata_summary,
    _tensor_summary,
    decode_real_attention_projection_weights,
    deterministic_attention_activation,
    layer_attention_projection_keys,
)
from models.demos.deepseek_v4_flash.real_checkpoint_loader import (
    RealCheckpointTensorIndex,
    TensorMetadata,
    layer_shared_expert_mlp_keys,
)
from models.demos.deepseek_v4_flash.real_kv_projection_smoke import (
    DEFAULT_KV_PROJECTION_MAX_BYTES,
    DEFAULT_KV_PROJECTION_MAX_TENSORS,
    KvProjectionWeights,
    decode_real_kv_projection_weights,
    layer_kv_projection_keys,
)
from models.demos.deepseek_v4_flash.real_shared_expert_smoke import (
    DEFAULT_SHARED_EXPERT_MAX_BYTES,
    DEFAULT_SHARED_EXPERT_MAX_TENSORS,
    SHARED_EXPERT_TTNN_TILE_MULTIPLE,
    decode_real_shared_expert_weights,
)
from models.demos.deepseek_v4_flash.ttnn_attention_projection import AttentionProjectionWeights, TtAttentionProjection
from models.demos.deepseek_v4_flash.ttnn_shared_expert import TtSharedExpertMLP

REAL_TRACEABLE_DECODE_SMOKE_SCHEMA_VERSION = 1
DEFAULT_TRACEABLE_DECODE_LAYER = 3
DEFAULT_TRACEABLE_DECODE_SEQ_LEN = 32
DEFAULT_TRACEABLE_DECODE_CACHE_LEN = 64
DEFAULT_TRACEABLE_DECODE_MAX_TENSORS = (
    DEFAULT_ATTENTION_PROJECTION_MAX_TENSORS + DEFAULT_KV_PROJECTION_MAX_TENSORS + DEFAULT_SHARED_EXPERT_MAX_TENSORS + 1
)
DEFAULT_TRACEABLE_DECODE_MAX_BYTES = (
    DEFAULT_ATTENTION_PROJECTION_MAX_BYTES + DEFAULT_KV_PROJECTION_MAX_BYTES + DEFAULT_SHARED_EXPERT_MAX_BYTES + 4096
)
DEFAULT_TRACEABLE_DECODE_TRACE_REGION_SIZE = 64 * 1024 * 1024


@dataclass(frozen=True)
class TraceableDecodeWeights:
    attention: AttentionProjectionWeights
    kv: KvProjectionWeights
    attn_norm: torch.Tensor
    ffn_norm: torch.Tensor
    shared_expert: dict[str, torch.Tensor]


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
    append, and shared-expert FFN stepping stone:
    ``hidden -> attn_norm -> wq_a -> q_norm -> wq_b``,
    ``attn_norm -> wkv -> kv_norm -> update_cache``, and
    ``hidden -> ffn_norm -> shared expert -> residual``.
    """

    def __init__(
        self,
        *,
        device,
        weights: TraceableDecodeWeights,
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
        self.attention = TtAttentionProjection(
            device=device,
            weights=weights.attention,
            hidden_size=int(config.hidden_size),
            q_lora_rank=int(config.q_lora_rank),
            num_heads=int(config.num_attention_heads),
            head_dim=int(config.head_dim),
            norm_eps=float(config.rms_norm_eps),
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
        kv_update = ttnn.to_memory_config(
            kv_output,
            _kv_update_memory_config(device=self.device, token_rows=int(kv_output.shape[-2]), width=self.kv_output_dim),
        )
        self.kv_cache = ttnn.update_cache(self.kv_cache, kv_update, self.cache_update_index)
        ffn_norm_output = ttnn.rms_norm(
            hidden_states,
            weight=self.ffn_norm,
            epsilon=float(self.config.rms_norm_eps),
            memory_config=self.memory_config,
        )
        shared_output = self.shared_expert(ffn_norm_output)
        residual_output = ttnn.add(hidden_states, shared_output, memory_config=self.memory_config)
        return {
            "attn_norm_output": attn_norm_output,
            "q_rank_norm": q_rank_norm,
            "q_output": q_output,
            "kv_linear": kv_linear,
            "kv_output": kv_output,
            "kv_cache": self.kv_cache,
            "ffn_norm_output": ffn_norm_output,
            "shared_output": shared_output,
            "residual_output": residual_output,
        }


def run_traceable_decode_subpath_smoke(
    snapshot_dir: str | Path,
    *,
    layer: int = DEFAULT_TRACEABLE_DECODE_LAYER,
    seq_len: int = DEFAULT_TRACEABLE_DECODE_SEQ_LEN,
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
    replay_activation = _replay_activation(activation)
    reference = build_torch_traceable_decode_subpath_reference(
        weights,
        config=config,
        activation=activation,
        cache_len=cache_len,
        cache_update_index=cache_update_index,
    )
    replay_reference = build_torch_traceable_decode_subpath_reference(
        weights,
        config=config,
        activation=replay_activation,
        cache_len=cache_len,
        cache_update_index=cache_update_index,
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
        replay_activation=replay_activation,
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
        config=config,
        activation=activation,
        replay_activation=replay_activation,
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
        "ttnn.rms_norm(ffn_norm)",
        "ttnn.linear(shared_w1)",
        "ttnn.linear(shared_w3)",
        "ttnn.mul(silu(shared_gate),shared_up)",
        "ttnn.linear(shared_w2)",
        "ttnn.add(hidden,shared_output)",
    ]
    result["ttnn"] = {name: _tensor_summary(value) for name, value in ttnn_outputs.items()}
    result["accuracy"] = {
        name: _accuracy_summary(expected, ttnn_outputs[name], pcc_threshold=pcc, rtol=rtol, atol=atol)
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
    max_tensors: int = DEFAULT_TRACEABLE_DECODE_MAX_TENSORS,
    max_bytes: int = DEFAULT_TRACEABLE_DECODE_MAX_BYTES,
) -> tuple[dict[str, torch.Tensor], list[TensorMetadata], dict[str, list[str]]]:
    index = RealCheckpointTensorIndex.from_snapshot(snapshot_dir)
    attention_keys = layer_attention_projection_keys(index, layer=layer)
    kv_keys = [key for key in layer_kv_projection_keys(index, layer=layer) if key not in attention_keys]
    ffn_norm_keys = [f"layers.{layer}.ffn_norm.weight"]
    for key in ffn_norm_keys:
        index.location(key)
    shared_expert_keys = layer_shared_expert_mlp_keys(index, layer=layer)
    keys = {
        "attention_query": attention_keys,
        "kv_projection": kv_keys,
        "ffn_norm": ffn_norm_keys,
        "shared_expert": shared_expert_keys,
    }
    tensors, metadata = index.load_tensors(
        _unique_keys([*attention_keys, *kv_keys, *ffn_norm_keys, *shared_expert_keys]),
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    return tensors, metadata, keys


def decode_traceable_decode_subpath_weights(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
) -> TraceableDecodeWeights:
    attention = decode_real_attention_projection_weights(tensors, config=config, layer=layer)
    kv = decode_real_kv_projection_weights(tensors, config=config, layer=layer)
    ffn_norm_key = f"layers.{layer}.ffn_norm.weight"
    if ffn_norm_key not in tensors:
        raise KeyError(f"Missing required FFN norm tensor {ffn_norm_key!r}")
    if tuple(tensors[ffn_norm_key].shape) != (int(config.hidden_size),):
        raise ValueError(
            f"Expected {ffn_norm_key} shape {(int(config.hidden_size),)}, " f"got {tuple(tensors[ffn_norm_key].shape)}"
        )
    shared_expert = decode_real_shared_expert_weights(tensors, config=config, layer=layer)
    return TraceableDecodeWeights(
        attention=attention,
        kv=kv,
        attn_norm=tensors[f"layers.{layer}.attn_norm.weight"].contiguous().to(torch.bfloat16),
        ffn_norm=tensors[ffn_norm_key].contiguous().to(torch.bfloat16),
        shared_expert=shared_expert,
    )


def build_torch_traceable_decode_subpath_reference(
    weights: TraceableDecodeWeights,
    *,
    config: DeepSeekV4FlashConfig,
    activation: torch.Tensor,
    cache_len: int,
    cache_update_index: int,
) -> dict[str, torch.Tensor]:
    _validate_activation(activation, hidden_size=int(config.hidden_size))
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

    ffn_norm_output = rms_norm(
        activation[:, 0],
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
    residual_output = (activation.float() + shared_output.float()).to(torch.bfloat16)
    return {
        "attn_norm_output": attn_norm_output.to(torch.bfloat16),
        "q_rank_norm": q_rank_norm.to(torch.bfloat16),
        "q_output": q_output.to(torch.bfloat16),
        "kv_linear": kv_linear.unsqueeze(1).to(torch.bfloat16),
        "kv_output": kv_output.to(torch.bfloat16),
        "kv_cache": kv_cache,
        "ffn_norm_output": ffn_norm_output.to(torch.bfloat16),
        "shared_output": shared_output.to(torch.bfloat16),
        "residual_output": residual_output,
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
            "models.demos.deepseek_v4_flash.ttnn_attention_projection",
            "TtAttentionProjection.project_output",
            "TtAttentionProjection.project_output(host_wo_a)",
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
    config: DeepSeekV4FlashConfig,
    activation: torch.Tensor,
    replay_activation: torch.Tensor,
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
        _copy_activation_to_device(activation, tt_input)
        module(tt_input)
        ttnn.synchronize_device(device)

        with TraceableDecodeHostGuard() as guard:
            trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            output_tensors = module(tt_input)
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
        ttnn.synchronize_device(device)

        _copy_activation_to_device(replay_activation, tt_input)
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
    replay_activation: torch.Tensor,
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
            "moe_intermediate_size": int(config.moe_intermediate_size),
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
                "ttnn.rms_norm(ffn_norm)",
                "TtSharedExpertMLP",
                "ttnn.add(hidden,shared_output)",
            ],
            "path": (
                "decode hidden state -> attn_norm/query projection plus K/V projection/cache append, and "
                "decode hidden state -> "
                "ffn_norm/shared expert/residual"
            ),
            "logical_decode_token_policy": (
                "the first token is the logical decode token; tensor shape is tile-padded/static for trace replay"
            ),
            "excluded_from_trace": [
                "K/V RoPE split and final sparse-attention cache read path",
                "host sparse-attention gather/softmax/reduction",
                "grouped wo_a attention output projection",
                "router scoring/top-k/hash selection",
                "routed expert gather/scatter/combine",
                "cache advancement beyond the fixed traced update index",
                "embedding and logits",
            ],
        },
        "selected_source_keys": [item.source_key for item in metadata],
        "loaded_tensors": [_metadata_summary(item) for item in metadata],
        "loaded_tensor_groups": loaded_groups,
        "loaded_real_tensor_groups": loaded_groups,
        "payload_bytes": {
            "attention_query": loaded_groups["attention_query"]["payload_bytes"],
            "kv_projection": loaded_groups["kv_projection"]["payload_bytes"],
            "ffn_norm": loaded_groups["ffn_norm"]["payload_bytes"],
            "shared_expert": loaded_groups["shared_expert"]["payload_bytes"],
            "total": sum(item.nbytes for item in metadata),
        },
        "decoded_tensors": {
            "attn_norm": _tensor_summary(weights.attn_norm),
            "q_norm": _tensor_summary(weights.attention.q_norm),
            "wq_a": _tensor_summary(weights.attention.wq_a),
            "wq_b": _tensor_summary(weights.attention.wq_b),
            "wkv": _tensor_summary(weights.kv.wkv),
            "kv_norm": _tensor_summary(weights.kv.kv_norm),
            "kv_cache_initial": _tensor_summary(torch.zeros((1, 1, int(cache_len), _kv_output_dim(config)))),
            "ffn_norm": _tensor_summary(weights.ffn_norm),
            "shared_w1": _tensor_summary(weights.shared_expert["w1"]),
            "shared_w2": _tensor_summary(weights.shared_expert["w2"]),
            "shared_w3": _tensor_summary(weights.shared_expert["w3"]),
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
                "name": "trace_output_readback",
                "location": "after trace replay",
                "description": "TTNN outputs are copied to host after replay for accuracy checks only",
            },
        ],
        "host_boundaries_inside_trace": [],
        "host_boundaries_outside_trace": [
            "real_weight_decode_to_bf16",
            "kv_cache_zero_init_host_to_device",
            "activation_host_to_device",
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
            "torch.rms_norm_reference(ffn_norm)",
            "torch.shared_swiglu_expert_reference",
            "torch.add(hidden,shared_output)",
        ],
        "ttnn_ops": [],
        "inputs": {
            "capture_activation": _tensor_summary(activation),
            "replay_activation": _tensor_summary(replay_activation),
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


def _replay_activation(activation: torch.Tensor) -> torch.Tensor:
    token_scale = torch.linspace(1.05, 0.95, steps=activation.shape[-2], dtype=torch.float32).reshape(1, 1, -1, 1)
    return (activation.float().flip(-2) * token_scale - 0.03125).to(torch.bfloat16).contiguous()


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


def _validate_smoke_args(
    *,
    layer: int,
    seq_len: int,
    max_tensors: int,
    max_bytes: int,
    trace_region_size: int,
    cache_len: int,
    cache_update_index: int | None,
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
    if not 0.0 <= pcc <= 1.0:
        raise ValueError(f"pcc must be in [0, 1], got {pcc}")


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
